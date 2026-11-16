// Mission 3 - QR Quishing scenarios
// label: 0 = Safe, 1 = Phishing, 2 = Suspicious
// All URLs are real and openable in India region

export const QR_SCENARIOS = [
  {
    id: 'upi-gpay-fake',
    title: 'UPI Cashback QR',
    icon: '💰',
    location: 'WhatsApp Group Message',
    description: 'A WhatsApp message says "Scan this GPay QR to receive ₹500 cashback from our Diwali offer." The QR was forwarded by an unknown contact.',
    url: 'https://www.google.com/search?q=upi+cashback+scam+india',
    label: 1,
    hint: 'Legitimate UPI cashback never comes via WhatsApp QR from unknown contacts. This redirects to a search page — in a real attack it would be a credential-harvesting page.',
  },
  {
    id: 'sbi-kyc-fake',
    title: 'SBI KYC QR',
    icon: '🏦',
    location: 'SMS from "SBI-KYC"',
    description: 'An SMS from "SBI-KYC" says your account will be blocked in 24 hours. It asks you to scan the QR to complete KYC verification immediately.',
    url: 'https://www.google.com/search?q=sbi+kyc+scam+fake+sms',
    label: 1,
    hint: 'SBI never sends KYC QR codes via SMS. The sender "SBI-KYC" is spoofed. Real SBI KYC is done only at branches or onlinesbi.com.',
  },
  {
    id: 'flipkart-coupon-fake',
    title: 'Flipkart Coupon QR',
    icon: '🎁',
    location: 'Telegram Forwarded Image',
    description: 'A Telegram message claims "Flipkart Big Billion Days — Scan to get ₹2000 off on your next order. Offer expires tonight!"',
    url: 'https://www.google.com/search?q=flipkart+fake+coupon+qr+scam',
    label: 1,
    hint: 'Flipkart never distributes coupons via Telegram QR codes. Urgency ("expires tonight") is a classic scam pressure tactic.',
  },
  {
    id: 'paytm-collect-fake',
    title: 'Paytm Collect Request QR',
    icon: '💸',
    location: 'Kirana Store Counter',
    description: 'A shopkeeper shows you a QR on their phone saying "Scan to pay ₹150". But the QR was recently changed and the merchant name looks different from the shop name.',
    url: 'https://www.google.com/search?q=paytm+collect+request+scam',
    label: 2,
    hint: 'Always verify the merchant name shown after scanning matches the actual shop. Scammers replace legitimate QRs with their own collect-request QRs.',
  },
  {
    id: 'irctc-legit',
    title: 'IRCTC Ticket Booking',
    icon: '🚂',
    location: 'IRCTC Booking Confirmation Email',
    description: 'Your IRCTC booking confirmation email has a QR code. Scanning it opens the official IRCTC website to view your PNR status.',
    url: 'https://www.irctc.co.in/nget/train-search',
    label: 0,
    hint: 'Official irctc.co.in domain with HTTPS. This is the real Indian Railways ticketing portal.',
  },
  {
    id: 'amazon-in-legit',
    title: 'Amazon India Order QR',
    icon: '📦',
    location: 'Amazon Delivery Package',
    description: 'A QR code printed on your Amazon India delivery package. Scanning it takes you to the official Amazon India orders page.',
    url: 'https://www.amazon.in/gp/your-account/order-history',
    label: 0,
    hint: 'Official amazon.in domain with HTTPS. Safe to open.',
  },
  {
    id: 'zomato-legit',
    title: 'Zomato Order Receipt QR',
    icon: '🍕',
    location: 'Zomato Delivery Receipt',
    description: 'The paper receipt from your Zomato food delivery has a QR to rate your experience and view the order summary.',
    url: 'https://www.zomato.com/india',
    label: 0,
    hint: 'Official zomato.com domain with HTTPS. Safe to open.',
  },
  {
    id: 'phonepe-legit',
    title: 'PhonePe Merchant QR',
    icon: '💳',
    location: 'Verified Merchant Counter',
    description: 'A PhonePe QR code at a verified merchant counter with the official PhonePe sticker and merchant name clearly printed below.',
    url: 'https://www.phonepe.com/en/',
    label: 0,
    hint: 'Official phonepe.com domain. Merchant name matches the shop. Safe to scan.',
  },
  {
    id: 'google-login-fake',
    title: 'Google Login QR',
    icon: '🔐',
    location: 'Phishing Email Attachment',
    description: 'An email claims your Google account needs re-verification. It contains a QR code saying "Scan to verify your identity securely."',
    url: 'https://www.google.com/search?q=google+account+phishing+qr+code',
    label: 1,
    hint: 'Google never asks you to verify your account by scanning a QR in an email. This is a credential-phishing attempt.',
  },
  {
    id: 'uidai-legit',
    title: 'Aadhaar e-KYC QR',
    icon: '🪪',
    location: 'Aadhaar Card (printed)',
    description: 'The QR code printed on the back of your physical Aadhaar card. Scanning it shows your encrypted demographic details for offline KYC.',
    url: 'https://myaadhaar.uidai.gov.in/',
    label: 0,
    hint: 'Official uidai.gov.in domain. The Aadhaar card QR is a standard offline KYC mechanism — safe.',
  },
]

export function getRandomQrScenarios(count = 6) {
  const phishing = QR_SCENARIOS.filter((s) => s.label === 1)
  const suspicious = QR_SCENARIOS.filter((s) => s.label === 2)
  const safe = QR_SCENARIOS.filter((s) => s.label === 0)

  const pick = (pool) => {
    const idx = Math.floor(Math.random() * pool.length)
    return pool.splice(idx, 1)[0]
  }

  const picked = []
  // Guarantee at least 2 safe, 2 phishing, 1 suspicious
  ;[safe, safe, phishing, phishing, suspicious].forEach((pool) => {
    if (pool.length && picked.length < count) picked.push(pick(pool))
  })

  const remaining = [...safe, ...phishing, ...suspicious]
  while (picked.length < count && remaining.length) {
    picked.push(pick(remaining))
  }

  return picked.sort(() => Math.random() - 0.5)
}