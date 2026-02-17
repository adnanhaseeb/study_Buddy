# StudyBuddy - Your Personal Study Assistant

## What is StudyBuddy?

StudyBuddy is a smart study tool that helps you learn from your documents. Upload your PDF files or text documents, and StudyBuddy will help you:

- **Ask questions** about your documents and get accurate answers
- **Create summaries** of long documents
- **Generate flashcards** for studying
- **Export everything** to use with other study apps

Think of it like having a smart tutor who has read all your materials and can answer any question about them!

## What You Need Before Starting

1. **Python installed** on your computer (version 3.10 or newer)
2. **An OpenRouter API key** (don't worry, we'll show you how to get one)
3. **Your study documents** (PDF or text files)

## Getting Your OpenRouter API Key

StudyBuddy uses OpenRouter to access AI models for understanding and answering your questions. Here's how to get your API key:

1. Go to [OpenRouter's website](https://openrouter.ai)
2. Create a free account
3. Go to "Keys" section in your dashboard
4. Click "Create Key"
5. Copy the key (it looks like: sk-or-v1-...)
6. Keep this key safe - you'll need it later!

## How to Set Up StudyBuddy

### Step 1: Download the Files
Download all the StudyBuddy files to a folder on your computer.

### Step 2: Install Required Programs
Open your computer's command prompt (or PowerShell on Windows) and type:

```
pip install -r requirements.txt
```

This downloads all the helper programs StudyBuddy needs.

### Step 3: Add Your OpenRouter Key
You have two options:

**Option A: Set it each time (easier)**
Before running StudyBuddy, type this in your command prompt:
```
set OPENAI_API_KEY=your_openrouter_key_here
```
(Replace "your_openrouter_key_here" with your actual OpenRouter key)

**Option B: Set it permanently**
Create a file called `.env` in your StudyBuddy folder and add:
```
OPENAI_API_KEY=your_openrouter_key_here
```

*Note: Even though we use OpenRouter, the variable name is still OPENAI_API_KEY for compatibility.*

## How to Use StudyBuddy

### Step 1: Put Your Documents Ready
Put all your PDF and text files in the `data` folder.

### Step 2: Process Your Documents
Open command prompt in your StudyBuddy folder and type:
```
python ingest.py --input_dir data/
```

This teaches StudyBuddy about your documents. You only need to do this once for each set of documents.

### Step 3: Start StudyBuddy
Type this command:
```
streamlit run app.py
```

Your web browser will open with StudyBuddy ready to use!

## How to Use the StudyBuddy Interface

Once StudyBuddy opens in your browser, you'll see several tabs:

### 1. Q&A Chat Tab
- Type any question about your documents
- Click "Ask" to get an answer
- StudyBuddy will tell you which document the answer came from

### 2. Summary Tab
- Click "Generate Summary" to get a short version of your documents
- Click "Generate Key Points" to get the most important information

### 3. Flashcards Tab
- Choose how many flashcards you want (5-50)
- Click "Generate Flashcards" to create study cards
- Download them as a CSV file to use in other apps

### 4. Export Tab
- Download your summaries and flashcards
- Get instructions for importing into Anki (a popular flashcard app)
- Export your question history

## What Each File Does

- **app.py** - The main program that creates the web interface
- **ingest.py** - Reads your documents and prepares them for StudyBuddy
- **rag.py** - The "brain" that answers questions and creates summaries
- **embeddings.py** - Converts your documents into a format AI can understand
- **vectorstore.py** - Stores and searches through your documents
- **requirements.txt** - List of helper programs needed
- **data/** folder - Where you put your study documents

## Common Problems and Solutions

### "No module named..." error
**Problem:** Missing required programs
**Solution:** Run `pip install -r requirements.txt` again

### "OpenAI API key not found" error
**Problem:** StudyBuddy can't find your OpenRouter API key
**Solution:** Make sure you set your OPENAI_API_KEY with your OpenRouter key (see Step 3 above)

### "No documents found" error
**Problem:** StudyBuddy can't find your documents
**Solution:** Make sure your PDF/text files are in the `data` folder and run `python ingest.py --input_dir data/` again

### StudyBuddy gives wrong answers
**Problem:** Documents weren't processed correctly
**Solution:** Delete the `vectorstore` folder and run `python ingest.py --input_dir data/` again

### App won't start
**Problem:** Port might be busy
**Solution:** Try `streamlit run app.py --server.port 8502`

## Tips for Best Results

1. **Use clear, specific questions** - Instead of "What's this about?", ask "What are the main causes of climate change?"

2. **Upload relevant documents** - Only upload documents related to what you're studying

3. **Break up large documents** - If you have a very long document, consider splitting it into chapters

4. **Check your questions** - If you get a weird answer, try rephrasing your question

## Need Help?

If something doesn't work:
1. Check that Python is installed correctly
2. Make sure your OpenRouter API key is set up correctly
3. Try running the commands again
4. Check that your documents are in the right folder

## What StudyBuddy Can and Cannot Do

**StudyBuddy CAN:**
- Answer questions about your uploaded documents
- Create summaries of your materials
- Generate flashcards for studying
- Export everything for other apps
- Remember your previous questions

**StudyBuddy CANNOT:**
- Access the internet for new information
- Answer questions about topics not in your documents  
- Translate languages (unless that info is in your documents)
- Do your homework for you (but it can help you understand!)

---

**Happy Studying!** 📚

StudyBuddy is designed to help you learn better, not to do the work for you. Use it as a study companion to understand your materials more deeply.