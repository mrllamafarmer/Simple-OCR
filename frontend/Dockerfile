FROM node:18

WORKDIR /app

# Copy package.json
COPY package.json ./

# Copy .env file
COPY .env ./

# Install only production dependencies
RUN npm install --only=production --ignore-scripts

# Copy the rest of the application code
COPY . .

# Build the app
RUN npm run build

# Install serve to run the built app
RUN npm install -g serve

# Expose the port the app runs on
EXPOSE 3300

# Start the app
CMD ["serve", "-s", "build", "-l", "3300"]