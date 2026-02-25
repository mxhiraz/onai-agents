# Photo Analysis Agent API

Express server that accepts an image and returns **pose**, **outfit**, or **studio** prompts (one comma-separated line each) using Google Gemini.

## Setup

```bash
bun install
```

Set your API key:

```bash
export GEMINI_API_KEY=your_key_here
```

## Run

```bash
bun start
```

Open **http://localhost:5001** in a browser to use the frontend: upload one image and get pose, outfit, and studio prompts in one go.

Dev with auto-reload:

```bash
bun run dev
```

## Docker

Create a `.env` file with your API key (or set it in the shell):

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

Then run with Docker Compose:

```bash
docker compose up --build
```

App and frontend: **http://localhost:5001**

## API

- **GET /health** — Health check.
- **POST /analyze/pose** — One-line pose prompt (body position, limbs, gaze).
- **POST /analyze/outfit** — One-line outfit prompt (clothing, styling, accessories).
- **POST /analyze/studio** — One-line studio prompt (lighting, background, atmosphere).

Send the image as multipart form data with field name `image`.

### Example (curl)

```bash
curl -X POST http://localhost:5001/analyze/pose -F "image=@/path/to/photo.jpg"
curl -X POST http://localhost:5001/analyze/outfit -F "image=@/path/to/photo.jpg"
curl -X POST http://localhost:5001/analyze/studio -F "image=@/path/to/photo.jpg"
```

Response is plain text: one comma-separated line.

### Example (JavaScript)

```js
const form = new FormData();
form.append('image', fileInput.files[0]);
const res = await fetch('http://localhost:5001/analyze/pose', {
  method: 'POST',
  body: form,
});
const text = await res.text();
console.log(text);
```
