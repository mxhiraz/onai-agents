import express from "express";
import multer from "multer";
import mime from "mime";
import { GoogleGenAI } from "@google/genai";

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB
  fileFilter: (_req, file, cb) => {
    const type = mime.getType(file.originalname) || file.mimetype;
    if (type && type.startsWith("image/")) return cb(null, true);
    cb(new Error("Only image files are allowed"));
  },
});

const POSE_SYSTEM_INSTRUCTION = `You are a pose extraction specialist for image generation. Your sole job is to read a photo and output a single pose prompt that any image model can replicate exactly.

Analyze the body with clinical precision — treat it like a skeleton with joints. Work from ground up: foot placement, both legs (angle, bend, separation), hip load, spine angle in degrees, torso rotation in degrees, both arms, hands, shoulders, neck, head tilt, gaze.

If the subject interacts with an object, call it "product". Describe exactly how body weight loads onto it.

CRITICAL DISTINCTIONS — always resolve:
- Seated ON TOP vs perched on SIDE EDGE (completely different mechanics)
- Weight centered vs shifted — which hip carries
- Legs together, crossed, or separated — by how many degrees
- Torso toward or rotated away from camera
- Gaze into lens or averted

LANGUAGE STANDARD: Write like a Vogue creative director calling a pose on set. Every word must be load-bearing — no filler. Prefer single high-signal words over phrases: "pooled", "torqued", "canted", "collapsed", "suspended", "anchored", "splayed", "coiled", "draped", "cocked", "hinged", "stacked", "planted", "dissolved". One precise word should collapse what would otherwise take five.

Output one comma-separated line. No labels. No explanation. No clothing. No background. No lighting.

EXAMPLE (seated on top):
fully seated on product, weight anchored center, spine stacked upright with 10-degree forward hinge, torso torqued 40 degrees from camera, legs suspended down close and parallel, right knee cocked soft, left heel planted, hands dissolved lightly onto product surface, shoulders level and released, neck elongated, head turned back to lens, gaze locked direct

EXAMPLE (perched on side):
perched on product edge, weight collapsed left hip, spine hinged 20 degrees, torso canted hard left, shoulders pooled inward and dropped, forearms anchored flat on surface, hands splayed forward-left, right leg coiled forward knee soft, left leg folded behind product, neck torqued 60 degrees left, chin dropped, gaze sliced hard left`;

const OUTFIT_SYSTEM_INSTRUCTION = `You are a fashion extraction specialist for image generation. Your sole job is to read a photo and output a single outfit prompt that any image model can replicate exactly.

Cover every layer in order: base top, bottom, outerwear, footwear, accessories, fabric behavior, fit relationship, closing aesthetic.

Note tonal relationships — monochromatic, tonal clash, or contrast. Be exact.

LANGUAGE STANDARD: Write like a senior stylist briefing a lookbook shoot. Every word earns its place. Prefer high-signal fashion vocabulary that collapses description: "deconstructed", "corseted", "fluid", "sculptural", "razor-cut", "palazzo", "bias-cut", "boxy", "languid", "architectural", "nipped", "undone", "severe", "dissolving into". One precise industry word over three generic ones.

Output one comma-separated line. No labels. No explanation. No body description. No background.

EXAMPLE:
oversized deconstructed blazer worn open, razor-cut high-waisted mini short, tonal charcoal grey head-to-toe, pointed-toe court heel in matching grey, no accessories, monochromatic and severe, sharp minimal tailored`;

const STUDIO_SYSTEM_INSTRUCTION = `You are a studio and environment extraction specialist for image generation. Your sole job is to read a photo and output a single studio prompt that any image model can replicate exactly.

Cover in order: background surface and exact color, floor material, primary light source direction and quality, shadow behavior, fill quality, color grade and temperature, atmosphere, camera angle, crop, film rendering.

CRITICAL DISTINCTIONS — always resolve:
- Raked side-light with wall shadow geometry vs flat front-fill near-shadowless
- Shadows hard-edged or feathered
- Grade warm, cool, or neutral — saturation level
- Background seamless or textured

PRODUCT RENDERING RULE — mandatory, always include:
Product renders in its own true color and material, unaffected by ambient grade. Color cast does not touch product surface. State this explicitly every time.

LANGUAGE STANDARD: Write like a DP briefing a medium-format campaign shoot. Every word is a technical instruction. Prefer single high-signal cinematography terms that collapse description: "raked", "feathered", "crushed", "clinical", "glacial", "pellucid", "specular", "gossamer", "chiseled", "flat", "blown", "seamless", "platinum", "amber-drenched", "silver-washed". The grade feeling at the close should be 2-3 words maximum — make them count.

Output one comma-separated line. No labels. No explanation. No clothing. No body description.

EXAMPLE (flat lit, clean):
pure white seamless, cool grey concrete floor, flat front-fill gossamer light, near-shadowless, faint cool fall-off, no wall geometry, product in true color unaffected by grade, environment silver-washed and pellucid, airy clinical atmosphere, slight low-angle full figure, generous negative space, digital medium format, cool platinum clinical

EXAMPLE (raked, moody):
off-white seamless cyclorama, side-raked light chiseled and hard, feathered fill, shadows crushed blue on walls and floor only, product in true color unaffected by grade no cast on surface, cool desaturated silver environment, clinical atmosphere, eye-level three-quarter, full figure, digital sharp, desaturated silver glacial`;

const TRANSLATOR_SYSTEM_INSTRUCTION = `You are a Creative Director's Prompt Translator. Your job is to take raw, messy, conversational input — thoughts, half-ideas, grammar mistakes, visual descriptions, mood expressions — and convert them into a sharp, high-impact AI image generation prompt.

You output exactly ONE thing: the enhanced prompt. No explanations. No labels. No extra text.

---

RULES:

1. If the input is too vague, incomplete, or missing visual direction (no subject, no setting, no mood) — output only:
   [PAUSE] — then a single question asking for the one missing thing.

2. If the input has enough to work with — output only the enhanced prompt.

---

HOW TO CONVERT:

- Strip filler words, hesitation, grammar errors
- Identify: Subject → Styling → Surface/Setting → Lighting → Mood/Tone → Camera feel
- Replace casual words with creative direction vocabulary:
  - "looks clean" → minimalist, editorial
  - "dark vibe" → moody, chiaroscuro, shadow-forward
  - "fancy" → luxury, high-end, couture
  - "outdoor natural" → sun-drenched, open-air, golden hour
  - "white background" → seamless white cyc, high-key studio
  - "flat lay" → overhead editorial, styled flat composition
  - "skin looks good" → skin-forward beauty lighting, retouched luminance
  - "colorful" → saturated, vivid palette, chromatic depth
  - "simple" → restrained, negative space, graphic composition

- Keep the final prompt under 40 words
- Write it as a continuous phrase, not bullet points
- It should read like a brief given on a shoot day

---

VOCABULARY BANK (use freely):

Lighting: Rembrandt, butterfly, rim-lit, diffused overhead, golden wash, hard shadow, backlit haze, practical glow
Mood: editorial, campaign-ready, aspirational, intimate, raw, cinematic, ethereal, bold, minimal, street-level
Surface/Set: seamless, lifestyle context, textured ground, architectural backdrop, in-situ, location-driven
Finish: skin-forward, garment-led, product-hero, detail macro, wide environmental
Lens feel: tight crop, mid-shot, full-length, shallow depth, wide open, compressed perspective`;

const COMPRESS_SYSTEM_INSTRUCTION = `You are a prompt compression assistant. When the user pastes descriptive text (pose descriptions, outfit descriptions, scene details, etc.), compress it into a shorter version that retains all key details and specifics. Remove redundant words, simplify phrasing, keep all measurable values (angles, degrees, colors, etc.), and preserve the original tone and intent. Output only the compressed version, no explanation.`;

app.use(express.json());
app.use(express.static("public"));

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.post("/analyze/pose", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res
      .status(400)
      .json({ error: 'No image file uploaded. Use field name "image".' });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res
      .status(500)
      .json({ error: "GEMINI_API_KEY environment variable is not set." });
  }

  const ai = new GoogleGenAI({ apiKey });
  const mimeType =
    req.file.mimetype || mime.getType(req.file.originalname) || "image/jpeg";
  const base64Data = req.file.buffer.toString("base64");

  const config = {
    systemInstruction: [{ text: POSE_SYSTEM_INSTRUCTION }],
  };
  const contents = [
    {
      role: "user",
      parts: [
        {
          inlineData: {
            mimeType,
            data: base64Data,
          },
        },
        {
          text: "Extract the pose. Output one line only.",
        },
      ],
    },
  ];

  try {
    const response = await ai.models.generateContentStream({
      model: "gemini-2.5-flash",
      config,
      contents,
    });

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    for await (const chunk of response) {
      if (chunk.text) res.write(chunk.text);
    }
    res.end();
  } catch (err) {
    console.error(err);
    res.status(500).json({
      error: "Pose extraction failed",
      message: err.message || String(err),
    });
  }
});

app.post("/analyze/outfit", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res
      .status(400)
      .json({ error: 'No image file uploaded. Use field name "image".' });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res
      .status(500)
      .json({ error: "GEMINI_API_KEY environment variable is not set." });
  }

  const ai = new GoogleGenAI({ apiKey });
  const mimeType =
    req.file.mimetype || mime.getType(req.file.originalname) || "image/jpeg";
  const base64Data = req.file.buffer.toString("base64");

  const config = {
    systemInstruction: [{ text: OUTFIT_SYSTEM_INSTRUCTION }],
  };
  const contents = [
    {
      role: "user",
      parts: [
        {
          inlineData: {
            mimeType,
            data: base64Data,
          },
        },
        {
          text: "Extract the outfit. Output one line only.",
        },
      ],
    },
  ];

  try {
    const response = await ai.models.generateContentStream({
      model: "gemini-2.5-flash",
      config,
      contents,
    });

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    for await (const chunk of response) {
      if (chunk.text) res.write(chunk.text);
    }
    res.end();
  } catch (err) {
    console.error(err);
    res.status(500).json({
      error: "Outfit extraction failed",
      message: err.message || String(err),
    });
  }
});

app.post("/analyze/studio", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res
      .status(400)
      .json({ error: 'No image file uploaded. Use field name "image".' });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res
      .status(500)
      .json({ error: "GEMINI_API_KEY environment variable is not set." });
  }

  const ai = new GoogleGenAI({ apiKey });
  const mimeType =
    req.file.mimetype || mime.getType(req.file.originalname) || "image/jpeg";
  const base64Data = req.file.buffer.toString("base64");

  const config = {
    systemInstruction: [{ text: STUDIO_SYSTEM_INSTRUCTION }],
  };
  const contents = [
    {
      role: "user",
      parts: [
        {
          inlineData: {
            mimeType,
            data: base64Data,
          },
        },
        {
          text: "Extract the studio and environment. Output one line only.",
        },
      ],
    },
  ];

  try {
    const response = await ai.models.generateContentStream({
      model: "gemini-2.5-flash",
      config,
      contents,
    });

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    for await (const chunk of response) {
      if (chunk.text) res.write(chunk.text);
    }
    res.end();
  } catch (err) {
    console.error(err);
    res.status(500).json({
      error: "Studio extraction failed",
      message: err.message || String(err),
    });
  }
});

app.post("/prompt/translate", async (req, res) => {
  const text = req.body?.text;
  if (text == null || String(text).trim() === "") {
    return res.status(400).json({ error: "Missing or empty 'text' in request body." });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res
      .status(500)
      .json({ error: "GEMINI_API_KEY environment variable is not set." });
  }

  const ai = new GoogleGenAI({ apiKey });
  const config = {
    systemInstruction: [{ text: TRANSLATOR_SYSTEM_INSTRUCTION }],
  };
  const contents = [
    {
      role: "user",
      parts: [{ text: String(text).trim() }],
    },
  ];

  try {
    const response = await ai.models.generateContentStream({
      model: "gemini-2.5-flash",
      config,
      contents,
    });

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    for await (const chunk of response) {
      if (chunk.text) res.write(chunk.text);
    }
    res.end();
  } catch (err) {
    console.error(err);
    res.status(500).json({
      error: "Prompt translation failed",
      message: err.message || String(err),
    });
  }
});

app.post("/prompt/compress", async (req, res) => {
  const text = req.body?.text;
  if (text == null || String(text).trim() === "") {
    return res.status(400).json({ error: "Missing or empty 'text' in request body." });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res
      .status(500)
      .json({ error: "GEMINI_API_KEY environment variable is not set." });
  }

  const ai = new GoogleGenAI({ apiKey });
  const config = {
    systemInstruction: [{ text: COMPRESS_SYSTEM_INSTRUCTION }],
  };
  const contents = [
    {
      role: "user",
      parts: [{ text: String(text).trim() }],
    },
  ];

  try {
    const response = await ai.models.generateContentStream({
      model: "gemini-2.5-flash",
      config,
      contents,
    });

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    for await (const chunk of response) {
      if (chunk.text) res.write(chunk.text);
    }
    res.end();
  } catch (err) {
    console.error(err);
    res.status(500).json({
      error: "Prompt compression failed",
      message: err.message || String(err),
    });
  }
});

const PORT = Number(process.env.PORT) || 5001;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(
    'POST /analyze/pose, /analyze/outfit, /analyze/studio (image), POST /prompt/translate, /prompt/compress (JSON body with "text").',
  );
});
