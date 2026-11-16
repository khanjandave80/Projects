import { GAME_EMAILS } from './gameEmailsData.js'

export function getRandomFallbackEmails(count = 10) {
  const phishing = GAME_EMAILS.filter((e) => e.label === 1)
  const safe = GAME_EMAILS.filter((e) => e.label === 0)

  const pick = (pool, n) => {
    const copy = [...pool]
    const out = []
    while (out.length < n && copy.length) {
      const i = Math.floor(Math.random() * copy.length)
      out.push(copy.splice(i, 1)[0])
    }
    return out
  }

  const half = Math.floor(count / 2)
  const picked = [...pick(phishing, half), ...pick(safe, count - half)]
  return picked.sort(() => Math.random() - 0.5)
}