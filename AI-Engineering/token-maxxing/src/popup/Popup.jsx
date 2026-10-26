import { useEffect, useRef, useState } from 'react'

import './Popup.css'

export const Popup = () => {
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const conversation = useRef(null)

  useEffect(() => {
    conversation.current.scrollTop = conversation.current.scrollHeight
  }, [messages])

  const send = async (event) => {
    event.preventDefault()
    const text = prompt.trim()
    if (!text || busy) return

    setBusy(true)
    setError('')
    try {
      const response = await fetch('http://127.0.0.1:8000/compress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text }),
      })
      if (!response.ok) throw new Error(`Compressor returned HTTP ${response.status}`)
      const { compressed_prompt } = await response.json()
      if (typeof compressed_prompt !== 'string') throw new Error('Invalid compressor response')

      setMessages((current) => [...current, { role: 'you', text }, { role: 'echo', text: compressed_prompt }])
      setPrompt('')
    } catch (failure) {
      setError(`${failure.message}. Check that the local compressor is running.`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="chat">
      <header className="chat-header">
        <span className="chat-mark" aria-hidden="true">✳</span>
        <div>
          <h1>Token Maxxing</h1>
          <p>Compressed echo</p>
        </div>
      </header>

      <section className="messages" aria-label="Conversation" aria-live="polite" ref={conversation}>
        {messages.length === 0 ? (
          <div className="empty-state">
            <span className="empty-icon" aria-hidden="true">✳</span>
            <h2>Start a conversation</h2>
            <p>Paste a prompt below to see a shorter version.</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div className={`message message-${message.role}`} key={index}>
              <span className="message-label">{message.role === 'you' ? 'You' : 'Echo'}</span>
              <p>{message.text}</p>
            </div>
          ))
        )}
      </section>

      {error && <p className="chat-error" role="alert">{error}</p>}
      <form className="composer" onSubmit={send}>
        <label className="sr-only" htmlFor="prompt">Message</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Paste a prompt..."
          rows="3"
          maxLength="20000"
          disabled={busy}
        />
        <button type="submit" disabled={!prompt.trim() || busy} aria-label="Send message">
          {busy ? 'Working…' : 'Send'}
        </button>
      </form>
    </main>
  )
}

export default Popup
