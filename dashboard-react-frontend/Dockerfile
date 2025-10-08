FROM node:20-alpine

WORKDIR /app

# cache de dependências
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm ci || npm install

# código
COPY . .

EXPOSE 5173
CMD ["sh", "-c", "npm run dev"]
