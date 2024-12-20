# MTAI_Python_Project

Telegram bot for classifying MNIST-like images using Redis for storage and indexing.

## Features
- `/init`: Initialize Redis index with MNIST data.
- `/add <label>`: Add a custom image to Redis.
- `/predict`: Predict the label of an image.

## Setup

### Clone the repository:
```bash
git clone git@github.com:Square596/MTAI_Python_Project.git
cd MTAI_Python_Project
```

### Add Telegram Bot Token
Create a .tg_bot_token file in the project root directory and add your bot's API token (from @BotFather).

### Build and start the containers:
```bash
docker-compose up --build
```

### Running the Bot
Once the containers are up, the bot will be live on Telegram. Use the following commands:

- `/start`: Start the bot.
- `/help`: Show help.
- `/init`: Initialize Redis index.
- `/add <label>`: Add an image to Redis.
- `/predict`: Predict the label of an image.

### Stop the Containers
To stop the containers:

```bash
docker-compose down
```
