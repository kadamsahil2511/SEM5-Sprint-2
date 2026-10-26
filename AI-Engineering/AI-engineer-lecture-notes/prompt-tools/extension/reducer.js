// reducer.js
// Token counting + prompt compression. Two paths:
//   - reduceRuleBased(text): fast, regex-driven.
//   - reduceNLP(text): deeper, POS-aware, uses compromise. 100% offline.
// Attached to `self` so it works in popup + content script contexts.

(function (root) {
  "use strict";

  // --- Token counting -------------------------------------------------------
  function countTokens(text) {
    if (!text) return 0;
    const nonLatin = /[^\u0000-ɏ\s]/.test(text);
    return nonLatin ? Math.ceil(text.length / 2) : Math.ceil(text.length / 4);
  }

  // --- Shared data ----------------------------------------------------------
  const POLITENESS = [
    "please", "kindly", "thanks", "thank you", "thanks in advance",
    "appreciate it", "if possible", "if you don't mind", "i'd love it if",
    "i was wondering if", "could you possibly", "would you mind"
  ];
  const HEDGES = [
    "basically", "essentially", "actually", "literally", "honestly",
    "really just", "i think", "i believe", "it seems", "kind of",
    "sort of", "somewhat"
  ];
  const ORNAMENTAL_INTENSIFIERS = ["very", "really", "quite", "pretty", "rather", "fairly"];
  const COMPARATIVE_NEXT = new Set([
    "first", "last", "best", "worst", "same", "only", "unique",
    "most", "least", "greatest", "nearest", "earliest", "latest"
  ]);

  const PHRASE_MAP = {
    "in order to": "to",
    "a large number of": "many",
    "a majority of": "most",
    "due to the fact that": "because",
    "in the event that": "if",
    "at this point in time": "now",
    "at the present time": "now",
    "for the purpose of": "to",
    "with regard to": "about",
    "with reference to": "about",
    "in spite of the fact that": "although",
    "make a decision": "decide",
    "give consideration to": "consider",
    "take into consideration": "consider",
    "has the ability to": "can",
    "is able to": "can",
    "is capable of": "can",
    "in the near future": "soon",
    "a sufficient amount of": "enough"
  };

  // Light-verb + nominalization patterns.
  // [light-verb] + [article] + [nominalization] + [preposition] → single verb.
  const NOMINALIZATION_MAP = {
    "make an improvement to": "improve",
    "make improvements to": "improve",
    "give a demonstration of": "demonstrate",
    "conduct an investigation into": "investigate",
    "conduct an investigation of": "investigate",
    "carry out an analysis of": "analyze",
    "perform a review of": "review",
    "provide an explanation of": "explain",
    "make an analysis of": "analyze",
    "provide a summary of": "summarize",
    "give a summary of": "summarize"
  };

  // IBM-tutorial-style token-level abbreviations (deterministic, lossless-ish).
  // Keys must be matched as whole words, case-insensitively.
  const ABBREVIATIONS = {
    "approximately": "~",
    "about": "~",
    "versus": "vs",
    "maximum": "max",
    "minimum": "min",
    "documentation": "docs",
    "document": "doc",
    "documents": "docs",
    "parameter": "param",
    "parameters": "params",
    "configuration": "config",
    "repository": "repo",
    "repositories": "repos",
    "environment": "env",
    "number of": "#",
    "percent": "%",
    "with": "w/",
    "without": "w/o",
    "and so on": "etc.",
    "for example": "e.g.",
    "that is": "i.e.",
    "seconds": "s",
    "minutes": "min",
    "hours": "h",
    "days": "d",
    "weeks": "w",
    "months": "mo",
    "years": "y"
  };

  const REDUNDANT_MODIFIERS = {
    "end": "result",
    "final": "outcome",
    "advance": "planning",
    "unexpected": "surprise",
    "basic": "fundamentals",
    "past": "history",
    "future": "plans",
    "exact": "same"
  };

  function escapeRegex(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  // --- Rule-based reducer (unchanged, kept for Light mode) ------------------
  const FILLER_PHRASES = [
    "i was wondering if you could", "i was wondering if",
    "if it's not too much trouble", "if possible",
    "thanks in advance", "thank you so much", "thank you", "thanks a lot", "thanks",
    "would you mind", "could you please", "could you kindly", "could you",
    "would you", "kindly", "please"
  ];
  const PHRASE_SWAPS = Object.entries(PHRASE_MAP).map(([k, v]) =>
    [new RegExp("\\b" + escapeRegex(k) + "\\b", "gi"), v]
  );

  function jaccard(a, b) {
    const wa = new Set(a.toLowerCase().match(/\w+/g) || []);
    const wb = new Set(b.toLowerCase().match(/\w+/g) || []);
    if (!wa.size && !wb.size) return 1;
    let inter = 0;
    wa.forEach(w => { if (wb.has(w)) inter++; });
    const union = wa.size + wb.size - inter;
    return union === 0 ? 0 : inter / union;
  }

  function reduceRuleBased(text) {
    const log = [];
    const before = countTokens(text);
    let out = text;

    out = out.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").replace(/ *\n */g, "\n").trim();

    for (const phrase of FILLER_PHRASES) {
      const re = new RegExp("\\b" + escapeRegex(phrase) + "\\b[,.!?]?\\s*", "gi");
      const m = out.match(re);
      if (m) { out = out.replace(re, ""); log.push(`removed "${phrase}" x${m.length}`); }
    }
    for (const [re, rep] of PHRASE_SWAPS) {
      const m = out.match(re);
      if (m) { out = out.replace(re, rep); log.push(`"${m[0]}" → "${rep}"`); }
    }
    // Token-level abbreviations (also helps non-prose prompts).
    for (const [long, short] of Object.entries(ABBREVIATIONS)) {
      const re = new RegExp("\\b" + escapeRegex(long) + "\\b", "gi");
      const m = out.match(re);
      if (m) { out = out.replace(re, short); log.push(`"${long}" → "${short}" x${m.length}`); }
    }

    const sentences = out.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
    const seen = new Set();
    const deduped = [];
    let dropped = 0;
    for (const s of sentences) {
      const key = s.toLowerCase();
      if (seen.has(key)) { dropped++; continue; }
      seen.add(key); deduped.push(s);
    }
    if (dropped) log.push(`dropped ${dropped} duplicate sentence(s)`);
    out = deduped.join(" ");

    const lines = out.split(/\n/);
    const bulletRe = /^\s*([-*•]|\d+[.)])\s+/;
    const kept = [];
    const bulletSeen = [];
    let bDrop = 0;
    for (const line of lines) {
      if (bulletRe.test(line)) {
        const body = line.replace(bulletRe, "");
        if (bulletSeen.find(b => jaccard(b, body) > 0.9)) { bDrop++; continue; }
        bulletSeen.push(body);
      }
      kept.push(line);
    }
    if (bDrop) log.push(`dropped ${bDrop} near-duplicate bullet(s)`);
    out = kept.join("\n");

    out = out.replace(/[ \t]{2,}/g, " ").replace(/\s+([,.!?])/g, "$1").trim();
    out = out.replace(/(^|[.!?]\s+)([a-z])/g, (_, p, c) => p + c.toUpperCase());

    const after = countTokens(out);
    return { reduced: out, savedTokens: before - after, log };
  }

  // --- NLP reducer (compromise-powered, fully offline) ----------------------
  // nlp is loaded globally via vendor/compromise.min.js before this file.
  function nlpLib() {
    const n = root.nlp || (typeof window !== "undefined" ? window.nlp : null);
    if (!n) throw new Error("compromise (nlp) not loaded.");
    return n;
  }

  // Rejoin an array of sentence strings into text, keeping blank lines
  // where consecutive sentences originally sat in different paragraphs.
  function joinSentences(parts) {
    return parts.filter(Boolean).join(" ");
  }

  function normalize(text) {
    // Smart quotes → straight.
    let s = text
      .replace(/[‘’′]/g, "'")
      .replace(/[“”″]/g, '"')
      .replace(/—/g, "--")
      .replace(/–/g, "-");
    // Preserve paragraph breaks: collapse 3+ newlines to 2.
    s = s.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n");
    // Collapse spaces/tabs on each line and trim per-line.
    s = s.split("\n").map(l => l.replace(/[ \t]+/g, " ").trim()).join("\n");
    return s.trim();
  }

  function splitParagraphs(text) {
    return text.split(/\n{2,}/);
  }

  function removeFillers(sentence) {
    // Case-insensitive multi-word phrase stripping with word boundaries.
    let s = sentence;
    for (const phrase of POLITENESS) {
      const re = new RegExp("\\b" + escapeRegex(phrase) + "\\b[,:]?\\s*", "gi");
      s = s.replace(re, "");
    }
    // Hedges: drop unless at the very start acting as subject ("I think ...").
    // We treat sentence-leading "I think/I believe/It seems" as subject and keep.
    for (const hedge of HEDGES) {
      const re = new RegExp("\\b" + escapeRegex(hedge) + "\\b[,:]?\\s*", "gi");
      // If sentence starts with the hedge (subject position), keep it.
      const leading = new RegExp("^\\s*" + escapeRegex(hedge) + "\\b", "i");
      if (leading.test(s)) continue;
      s = s.replace(re, "");
    }
    // Ornamental intensifiers before non-comparative adjectives.
    for (const intens of ORNAMENTAL_INTENSIFIERS) {
      const re = new RegExp("\\b" + escapeRegex(intens) + "\\s+(\\w+)", "gi");
      s = s.replace(re, (match, next) => {
        if (COMPARATIVE_NEXT.has(next.toLowerCase())) return match;
        return next;
      });
    }
    return s.replace(/\s{2,}/g, " ").replace(/\s+([,.!?])/g, "$1").trim();
  }

  function phraseSwap(sentence) {
    const nlp = nlpLib();
    let doc = nlp(sentence);
    for (const [verbose, concise] of Object.entries(PHRASE_MAP)) {
      if (doc.has(verbose)) doc.match(verbose).replaceWith(concise);
    }
    return doc.out("text");
  }

  function nominalizationReversal(sentence) {
    const nlp = nlpLib();
    let doc = nlp(sentence);
    for (const [pattern, concise] of Object.entries(NOMINALIZATION_MAP)) {
      if (doc.has(pattern)) doc.match(pattern).replaceWith(concise);
    }
    return doc.out("text");
  }

  // IBM token-level optimization: abbreviate common long-form tokens.
  // Applied with word boundaries, case-insensitively. Guards against matching
  // inside already-abbreviated or compound tokens.
  function abbreviate(sentence) {
    let s = sentence;
    for (const [long, short] of Object.entries(ABBREVIATIONS)) {
      const re = new RegExp("\\b" + escapeRegex(long) + "\\b", "gi");
      s = s.replace(re, short);
    }
    return s;
  }

  // Function-word pruning in Deep mode. LLMLingua-style: content words carry
  // most of the information; drop low-information fillers where safe.
  // Only drops standalone articles/determiners between content words, not at
  // sentence start (where removal would read wrong).
  // Stopwords that carry little information in most prompts. Conservative set:
  // articles, copulas, common prepositions, and auxiliaries. "not" is NEVER
  // dropped (meaning-flipping). Pronouns and "to" are kept (often critical).
  const STOPWORDS = new Set([
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "of", "with", "for", "in", "on", "at", "by", "from",
    "that", "which",
    "will", "would", "shall", "should", "may", "might", "can", "could"
  ]);
  function dropStopwords(sentence) {
    const tokens = sentence.split(/(\s+)/); // preserve whitespace positions
    const out = [];
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      // Only consider pure word tokens; skip whitespace and punctuation-bearing bits.
      const bare = t.toLowerCase().replace(/[^a-z']/g, "");
      const isWord = /[a-z]/.test(t) && !/[.!?,:;]/.test(t);
      if (isWord && STOPWORDS.has(bare) && out.length > 0) {
        if (i + 1 < tokens.length && /^\s+$/.test(tokens[i + 1])) i++;
        continue;
      }
      out.push(t);
    }
    return out.join("").replace(/\s{2,}/g, " ").trim();
  }

  function redundantModifiers(sentence) {
    const nlp = nlpLib();
    let doc = nlp(sentence);
    for (const [adj, noun] of Object.entries(REDUNDANT_MODIFIERS)) {
      const pattern = `${adj} ${noun}`;
      if (doc.has(pattern)) doc.match(pattern).replaceWith(noun);
    }
    return doc.out("text");
  }

  // Content-word lemma vector for a sentence.
  function contentLemmas(sentence) {
    const nlp = nlpLib();
    const doc = nlp(sentence);
    const set = new Set();
    doc.nouns().forEach(t => { const v = t.out("normal"); if (v) set.add(v); });
    doc.verbs().forEach(t => { const v = t.out("normal"); if (v) set.add(v); });
    doc.adjectives().forEach(t => { const v = t.out("normal"); if (v) set.add(v); });
    return set;
  }

  function jaccardSet(a, b) {
    if (!a.size && !b.size) return 1;
    let inter = 0;
    a.forEach(w => { if (b.has(w)) inter++; });
    const union = a.size + b.size - inter;
    return union === 0 ? 0 : inter / union;
  }

  function dedupeSentences(sentences, logLines) {
    const vecs = sentences.map(contentLemmas);
    const dropIdx = new Set();
    for (let i = 0; i < sentences.length; i++) {
      if (dropIdx.has(i)) continue;
      for (let j = i + 1; j < sentences.length; j++) {
        if (dropIdx.has(j)) continue;
        const sim = jaccardSet(vecs[i], vecs[j]);
        if (sim > 0.75) {
          // Keep the shorter one.
          if (sentences[i].length <= sentences[j].length) {
            dropIdx.add(j);
            logLines.push(`dedup: dropped "${sentences[j].slice(0, 50)}..." (sim ${sim.toFixed(2)})`);
          } else {
            dropIdx.add(i);
            logLines.push(`dedup: dropped "${sentences[i].slice(0, 50)}..." (sim ${sim.toFixed(2)})`);
            break;
          }
        }
      }
    }
    return sentences.filter((_, i) => !dropIdx.has(i));
  }

  function dedupeListItems(text, logLines) {
    const bulletRe = /^(\s*)([-*•]|\d+[.)])\s+(.*)$/;
    const lines = text.split("\n");
    const out = [];
    let currentBlock = null; // {start, items: [{idx, body, vec}]}

    const flush = () => {
      if (!currentBlock) return;
      const items = currentBlock.items;
      const drop = new Set();
      for (let i = 0; i < items.length; i++) {
        if (drop.has(i)) continue;
        for (let j = i + 1; j < items.length; j++) {
          if (drop.has(j)) continue;
          if (jaccardSet(items[i].vec, items[j].vec) > 0.75) {
            if (items[i].body.length <= items[j].body.length) {
              drop.add(j);
              logLines.push(`list dedup: dropped bullet "${items[j].body.slice(0, 40)}..."`);
            } else {
              drop.add(i);
              logLines.push(`list dedup: dropped bullet "${items[i].body.slice(0, 40)}..."`);
              break;
            }
          }
        }
      }
      for (let k = 0; k < items.length; k++) {
        if (!drop.has(k)) out.push(items[k].line);
      }
      currentBlock = null;
    };

    for (const line of lines) {
      const m = bulletRe.exec(line);
      if (m) {
        if (!currentBlock) currentBlock = { items: [] };
        currentBlock.items.push({ line, body: m[3], vec: contentLemmas(m[3]) });
      } else {
        flush();
        out.push(line);
      }
    }
    flush();
    return out.join("\n");
  }

  // Split a sentence into comma phrases, drop near-duplicates by content
  // lemmas (keeps the shorter one). Also exact-dedupes case-insensitively.
  function dedupeCommaPhrases(sentence, logLines) {
    if (!sentence || sentence.indexOf(",") === -1) return sentence;
    // Split keeping leading/trailing whitespace trimmed per piece.
    const parts = sentence.split(",").map(p => p.trim());
    if (parts.length < 3) return sentence;

    const vecs = parts.map(contentLemmas);
    const drop = new Set();
    const seenExact = new Map();
    for (let i = 0; i < parts.length; i++) {
      if (!parts[i]) { drop.add(i); continue; }
      const key = parts[i].toLowerCase();
      if (seenExact.has(key)) {
        drop.add(i);
        logLines.push(`phrase dedup: dropped exact dup "${parts[i].slice(0, 40)}"`);
        continue;
      }
      seenExact.set(key, i);
      for (let j = i + 1; j < parts.length; j++) {
        if (drop.has(j)) continue;
        if (vecs[i].size < 2 || vecs[j].size < 2) continue;
        if (jaccardSet(vecs[i], vecs[j]) > 0.8) {
          if (parts[i].length <= parts[j].length) {
            drop.add(j);
            logLines.push(`phrase dedup: dropped "${parts[j].slice(0, 40)}"`);
          } else {
            drop.add(i);
            logLines.push(`phrase dedup: dropped "${parts[i].slice(0, 40)}"`);
            break;
          }
        }
      }
    }
    const kept = parts.filter((_, i) => !drop.has(i));
    return kept.join(", ");
  }

  function collapseExamples(sentence, logLines) {
    const signals = /(e\.g\.|for example,?|such as)/i;
    const guard = /(all of|including the following)/i;
    if (!signals.test(sentence) || guard.test(sentence)) return sentence;
    // Find signal, slurp items separated by commas until a period/semicolon or end.
    const m = signals.exec(sentence);
    if (!m) return sentence;
    const start = m.index + m[0].length;
    const tail = sentence.slice(start);
    // Grab the span up to a terminator.
    const stopMatch = /[.;:]/.exec(tail);
    const span = stopMatch ? tail.slice(0, stopMatch.index) : tail;
    const items = span.split(",").map(s => s.trim()).filter(Boolean);
    if (items.length <= 5) return sentence;
    const trimmed = items.slice(0, 3).join(", ") + ", etc.";
    const rebuilt = sentence.slice(0, start) + " " + trimmed + (stopMatch ? tail.slice(stopMatch.index) : "");
    logLines.push(`collapsed example list (${items.length} → 3 + etc.)`);
    return rebuilt.replace(/\s{2,}/g, " ");
  }

  function instructionConsolidation(sentences, logLines) {
    // Pattern: imperative verb + first noun object → treat as key.
    const nlp = nlpLib();
    const seen = new Map();
    const kept = [];
    for (const s of sentences) {
      const doc = nlp(s);
      const verbs = doc.verbs().out("array");
      const nouns = doc.nouns().out("array");
      // Heuristic: imperative if sentence starts with a verb-like token.
      const first = (s.trim().split(/\s+/)[0] || "").toLowerCase().replace(/[^a-z']/g, "");
      const isImperative = verbs.length > 0 &&
        verbs[0].toLowerCase().replace(/[^a-z']/g, "") === first;
      if (isImperative && nouns.length > 0) {
        const key = (nlp(first).verbs().toInfinitive().out("normal") || first) + "::" + nouns[0].toLowerCase();
        if (seen.has(key)) {
          logLines.push(`consolidated repeated instruction: "${s.slice(0, 60)}"`);
          continue;
        }
        seen.set(key, true);
      }
      kept.push(s);
    }
    return kept;
  }

  function recapitalize(s) {
    return s.replace(/(^|[.!?]\s+)([a-z])/g, (_, p, c) => p + c.toUpperCase());
  }

  function tidy(s) {
    return s
      .replace(/\s+([,.!?;:])/g, "$1")
      .replace(/[ \t]{2,}/g, " ")
      .replace(/\(\s+/g, "(")
      .replace(/\s+\)/g, ")")
      .trim();
  }

  function reduceNLP(text) {
    const nlp = nlpLib();
    const totalBefore = countTokens(text);
    const breakdown = {
      "Normalization": 0,
      "Filler removal": 0,
      "Phrase swap": 0,
      "Nominalization": 0,
      "Redundant modifiers": 0,
      "Abbreviations": 0,
      "Stopword pruning": 0,
      "Dedup sentences": 0,
      "Dedup comma phrases": 0,
      "Dedup list items": 0,
      "Example collapse": 0,
      "Instruction consolidation": 0
    };
    const log = [];

    // STEP 1 — Normalize (paragraph preserving).
    const normalized = normalize(text);
    breakdown["Normalization"] = countTokens(text) - countTokens(normalized);

    // Process each paragraph independently so blank lines survive.
    const paragraphs = splitParagraphs(normalized);
    const outParagraphs = [];

    // For step 7/10 we want to pool sentences across the whole doc so repeats
    // in different paragraphs still consolidate. Collect first, filter later.
    let allSentences = [];
    const paraIndexForSentence = [];

    paragraphs.forEach((para, pIdx) => {
      const sents = nlp(para).sentences().out("array");
      sents.forEach(s => {
        allSentences.push(s);
        paraIndexForSentence.push(pIdx);
      });
    });

    // STEP 3 — Filler/hedge/intensifier removal.
    const beforeStep3 = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    allSentences = allSentences.map(removeFillers);
    const afterStep3 = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    breakdown["Filler removal"] = beforeStep3 - afterStep3;

    // STEP 4 — Phrase substitution.
    const beforeStep4 = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    allSentences = allSentences.map(phraseSwap);
    breakdown["Phrase swap"] = beforeStep4 - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // STEP 5 — Nominalization reversal.
    const beforeStep5 = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    allSentences = allSentences.map(nominalizationReversal);
    breakdown["Nominalization"] = beforeStep5 - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // STEP 6 — Redundant modifiers.
    const beforeStep6 = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    allSentences = allSentences.map(redundantModifiers);
    breakdown["Redundant modifiers"] = beforeStep6 - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // STEP 6a — Token-level abbreviations (IBM tutorial: days→d, docs, max, etc.).
    const beforeAbbr = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    allSentences = allSentences.map(abbreviate);
    breakdown["Abbreviations"] = beforeAbbr - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // STEP 6b — Stopword pruning (LLMLingua-style content retention: drop
    // articles, copulas, light prepositions and modals that carry little info).
    const beforeStop = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    allSentences = allSentences.map(dropStopwords);
    breakdown["Stopword pruning"] = beforeStop - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // STEP 9 — Collapse examples (per sentence; do before dedup so dedup compares
    // the already-trimmed version).
    const beforeStep9 = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    allSentences = allSentences.map(s => collapseExamples(s, log));
    breakdown["Example collapse"] = beforeStep9 - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // STEP 7 — Semantic sentence dedup.
    const beforeStep7 = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    // Keep paragraph index aligned.
    {
      const paired = allSentences.map((s, i) => ({ s, p: paraIndexForSentence[i] }));
      const strs = paired.map(x => x.s);
      const kept = dedupeSentences(strs, log);
      const keepSet = new Set(kept);
      const filtered = paired.filter(x => keepSet.has(x.s));
      // Rebuild aligned arrays.
      allSentences = filtered.map(x => x.s);
      paraIndexForSentence.length = 0;
      filtered.forEach(x => paraIndexForSentence.push(x.p));
    }
    breakdown["Dedup sentences"] = beforeStep7 - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // STEP 7b — Comma-phrase dedup (image/style prompts are often one long
    // comma-separated sentence; dedup repeated descriptors within each).
    const beforeStep7b = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    allSentences = allSentences.map(s => dedupeCommaPhrases(s, log));
    breakdown["Dedup comma phrases"] = beforeStep7b - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // STEP 10 — Instruction consolidation.
    const beforeStep10 = allSentences.map(countTokens).reduce((a, b) => a + b, 0);
    {
      const paired = allSentences.map((s, i) => ({ s, p: paraIndexForSentence[i] }));
      const strs = paired.map(x => x.s);
      const kept = instructionConsolidation(strs, log);
      const keepSet = new Set(kept);
      const filtered = paired.filter(x => keepSet.has(x.s));
      allSentences = filtered.map(x => x.s);
      paraIndexForSentence.length = 0;
      filtered.forEach(x => paraIndexForSentence.push(x.p));
    }
    breakdown["Instruction consolidation"] = beforeStep10 - allSentences.map(countTokens).reduce((a, b) => a + b, 0);

    // Reassemble into paragraphs.
    const byPara = {};
    allSentences.forEach((s, i) => {
      const p = paraIndexForSentence[i];
      if (!byPara[p]) byPara[p] = [];
      byPara[p].push(s);
    });
    for (let p = 0; p < paragraphs.length; p++) {
      const parts = byPara[p] || [];
      if (parts.length) outParagraphs.push(joinSentences(parts));
    }
    let reduced = outParagraphs.join("\n\n");

    // STEP 8 — Dedupe list items (operates on lines, so run after reassembly).
    const beforeStep8 = countTokens(reduced);
    reduced = dedupeListItems(reduced, log);
    breakdown["Dedup list items"] = beforeStep8 - countTokens(reduced);

    // STEP 11 — Final tidy.
    reduced = tidy(reduced);
    reduced = recapitalize(reduced);

    const totalAfter = countTokens(reduced);
    const savedTokens = totalBefore - totalAfter;

    // Prepend summary lines to log.
    const summary = Object.entries(breakdown)
      .filter(([, v]) => v !== 0)
      .map(([k, v]) => `${k}: ${v > 0 ? "-" : "+"}${Math.abs(v)} tokens`);

    return { reduced, savedTokens, log: summary.concat(log), breakdown };
  }

  root.PromptReducer = { countTokens, reduceRuleBased, reduceNLP };
})(typeof self !== "undefined" ? self : this);
