# dashboard-react-frontend/Dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

ENV VITE_API_URL=/api

# Se usar variáveis: VITE_API_BASE=/api (ou seu endpoint)
# ENV VITE_API_BASE=/api
RUN npm run build  # gera /dist

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Arquivo de config com SPA + proxy /api
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
