FROM oven/bun:1-alpine

WORKDIR /app

COPY package.json bun.lockb* package-lock.json* pnpm-lock.yaml* ./
RUN bun install

COPY . .

EXPOSE 5001

ENV NODE_ENV=production
ENV PORT=5001

CMD ["bun", "run", "server.js"]
