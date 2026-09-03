import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { extname, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
const dist = resolve(root, 'dist')
const serverDir = resolve(dist, 'server')
const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml; charset=utf-8',
}

async function collect(directory) {
  const files = []
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name)
    if (path === serverDir || entry.name === '.openai') continue
    if (entry.isDirectory()) files.push(...await collect(path))
    else files.push(path)
  }
  return files
}

const records = []
for (const path of await collect(dist)) {
  const url = `/${relative(dist, path).split(sep).join('/')}`
  const type = contentTypes[extname(path).toLowerCase()] || 'application/octet-stream'
  const binary = !type.includes('text') && !type.includes('json') && !type.includes('svg')
  const body = await readFile(path, binary ? undefined : 'utf8')
  records.push([url, [type, binary, binary ? body.toString('base64') : body]])
}

const worker = `const assets = new Map(${JSON.stringify(records)});\n` + String.raw`
function decodeBase64(value) {
  const raw = atob(value)
  const bytes = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)
  return bytes
}

export default {
  async fetch(request) {
    const url = new URL(request.url)
    let path = url.pathname === '/' ? '/index.html' : url.pathname
    let asset = assets.get(path)
    if (!asset && !path.includes('.')) asset = assets.get('/index.html')
    if (!asset) return new Response('Not found', { status: 404 })
    const [type, binary, body] = asset
    return new Response(binary ? decodeBase64(body) : body, {
      headers: {
        'Content-Type': type,
        'Cache-Control': path === '/index.html' ? 'no-cache' : 'public, max-age=31536000, immutable',
      },
    })
  },
}
`

await mkdir(serverDir, { recursive: true })
await writeFile(resolve(serverDir, 'index.js'), worker)
