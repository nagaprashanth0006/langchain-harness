FROM node:22-alpine AS build
WORKDIR /app
COPY nn-app/package.json nn-app/package-lock.json ./
RUN npm ci
COPY nn-app/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
