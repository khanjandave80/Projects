// Mission 2 - URL threat scenarios
// label: 0 = Safe, 1 = Phishing, 2 = Defacement, 3 = Malware

export const URL_SCENARIOS = [
  {
    id: 'bank-alert',
    title: 'Bank Account Suspended Alert',
    icon: '🏦',
    context: 'An SMS from "your bank" warns your account is suspended and asks you to click the link to verify your identity.',
    url: 'http://secure-bankofamerica-login.xyz/verify',
    label: 1,
    hint: 'Suspicious .xyz domain plus credential-verification path.',
  },
  {
    id: 'paypal-limited',
    title: 'PayPal Account Limited',
    icon: '💳',
    context: 'A message says your PayPal account is limited and you must update your details within 48 hours.',
    url: 'http://paypal-account-restore.tk/update',
    label: 1,
    hint: '.tk domain with a fake restore page.',
  },
  {
    id: 'sbi-kyc',
    title: 'SBI KYC Update',
    icon: '🏛️',
    context: 'A message claims your SBI KYC will expire soon and asks you to submit details via the link.',
    url: 'http://sbi-kyc-update.co/verify',
    label: 1,
    hint: 'Banks do not send KYC links from random lookalike domains.',
  },
  {
    id: 'amazon-gift',
    title: 'Amazon Gift Card Winner',
    icon: '🎁',
    context: 'A WhatsApp message says you won a ₹10,000 Amazon gift card and must claim it before it expires.',
    url: 'http://amazon-rewards-claim.ml/win?id=29471',
    label: 1,
    hint: 'Typosquatted Amazon branding plus a fake claim path.',
  },
  {
    id: 'news-deface',
    title: 'News Portal Defacement',
    icon: '📰',
    context: 'A news homepage suddenly shows altered branding and a rude takeover banner instead of normal content.',
    url: 'https://news-portal.example.com/index.html',
    label: 2,
    hint: 'Page content has been altered. This is a defacement case.',
  },
  {
    id: 'college-deface',
    title: 'College Website Defacement',
    icon: '🏫',
    context: 'A college website opens with a replacement splash screen and changed text that should not be there.',
    url: 'https://college-portal.example.edu/home',
    label: 2,
    hint: 'The website looks edited or replaced without authorization.',
  },
  {
    id: 'menu-apk',
    title: 'Restaurant Menu Download',
    icon: '🍽️',
    context: 'A café table sticker says "scan to download our menu app" for exclusive offers.',
    url: 'http://menu-download.tk/menu.apk',
    label: 3,
    hint: '.tk domain plus .apk file can hide a malicious payload.',
  },
  {
    id: 'free-tool',
    title: 'Free Tool Download',
    icon: '🛠️',
    context: 'A pop-up offers a free PDF converter but the download link looks suspicious.',
    url: 'http://free-tools-download.cf/tool.exe',
    label: 3,
    hint: 'Suspicious domain plus executable download.',
  },
  {
    id: 'cashback-app',
    title: 'Cashback Rewards App',
    icon: '💸',
    context: 'A forwarded ad claims a cashback app will pay you for scanning shopping receipts.',
    url: 'http://cashback-rewards-app.gq/app.zip',
    label: 3,
    hint: '.zip download from a risky domain is likely malware.',
  },
  {
    id: 'ransom-note',
    title: 'Fake Security Patch',
    icon: '🧯',
    context: 'A "critical patch" download claims to fix your device but behaves like a malware dropper.',
    url: 'http://critical-update-pack.xyz/patch.jar',
    label: 3,
    hint: 'A deceptive update package can deliver malicious code.',
  },
  {
    id: 'amazon-orders',
    title: 'Amazon Order Tracking',
    icon: '📦',
    context: 'Tracking link from Amazon for your recent delivery.',
    url: 'https://www.amazon.com/your-orders',
    label: 0,
    hint: 'Official amazon.com domain, safe to open.',
  },
  {
    id: 'hdfc-netbanking',
    title: 'HDFC NetBanking Login',
    icon: '🏦',
    context: 'Official HDFC Bank website to view your monthly statement.',
    url: 'https://www.hdfcbank.com/netbanking',
    label: 0,
    hint: 'Real hdfcbank.com domain with HTTPS.',
  },
  {
    id: 'google-signin',
    title: 'Google Account Sign-In',
    icon: '🔐',
    context: 'Google account password-change confirmation link.',
    url: 'https://accounts.google.com/signin',
    label: 0,
    hint: 'Official google.com domain, safe.',
  },
  {
    id: 'github',
    title: 'GitHub Repository',
    icon: '🐙',
    context: 'A link shared by a developer friend to a public repository.',
    url: 'https://github.com/octocat/Hello-World',
    label: 0,
    hint: 'Official github.com domain, safe.',
  },
  {
    id: 'zomato-orders',
    title: 'Zomato Order Page',
    icon: '🍕',
    context: 'Link from your Zomato order confirmation to view the order status.',
    url: 'https://www.zomato.com/orders',
    label: 0,
    hint: 'Official zomato.com domain, safe.',
  },
  {
    id: 'upi-pay',
    title: 'UPI Payment Link',
    icon: '💳',
    context: 'A trusted merchant shares a UPI payment link to settle your bill.',
    url: 'upi://pay?pa=merchant@okhdfcbank&pn=TrustedShop&am=250',
    label: 0,
    hint: 'Valid UPI payment scheme, safe.',
  },
]

export function getRandomUrlScenarios(count = 10) {
  const safe = URL_SCENARIOS.filter((s) => s.label === 0)
  const phishing = URL_SCENARIOS.filter((s) => s.label === 1)
  const defacement = URL_SCENARIOS.filter((s) => s.label === 2)
  const malware = URL_SCENARIOS.filter((s) => s.label === 3)

  const pick = (pool) => {
    const idx = Math.floor(Math.random() * pool.length)
    return pool.splice(idx, 1)[0]
  }

  const picked = []
  const pools = [safe, phishing, defacement, malware]

  pools.forEach((pool) => {
    if (pool.length && picked.length < count) {
      picked.push(pick(pool))
    }
  })

  const remaining = [...safe, ...phishing, ...defacement, ...malware]
  while (picked.length < count && remaining.length) {
    picked.push(pick(remaining))
  }

  return picked.sort(() => Math.random() - 0.5)
}