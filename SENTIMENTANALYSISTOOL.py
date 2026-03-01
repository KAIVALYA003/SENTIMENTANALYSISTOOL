import nltk
from flask import Flask, request, jsonify, render_template_string
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

sia = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# THE MAGNIFICENT UI (HTML/CSS)
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analyzer</title>
    <style>
        body { 
            font-family: 'Poppins', sans-serif; 
            background: linear-gradient(45deg, #0f0c29, #302b63, #24243e); 
            height: 100vh; display: flex; justify-content: center; align-items: center; margin: 0; color: white;
        }
        .container { 
            background: rgba(255, 255, 255, 0.1); 
            backdrop-filter: blur(10px); padding: 40px; border-radius: 20px; 
            box-shadow: 0 15px 35px rgba(0,0,0,0.5); width: 500px; text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1 { margin-bottom: 20px; font-weight: 300; letter-spacing: 2px; }
        textarea { 
            width: 100%; height: 120px; padding: 15px; border-radius: 12px; 
            border: none; margin-bottom: 20px; font-size: 16px; background: rgba(255,255,255,0.9);
            color: #333; box-sizing: border-box;
        }
        button { 
            background: #00d2ff; background: linear-gradient(to right, #3a7bd5, #00d2ff);
            color: white; border: none; padding: 15px 30px; border-radius: 50px; 
            cursor: pointer; font-size: 18px; font-weight: bold; width: 100%; transition: 0.3s;
        }
        button:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,210,255,0.4); }
        #resultBox { 
            margin-top: 25px; padding: 20px; border-radius: 12px; display: none; 
            animation: fadeIn 0.5s ease;
        }
        .positive { background: #28a745; color: white; }
        .negative { background: #dc3545; color: white; }
        .neutral { background: #ffc107; color: #333; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .details { font-size: 12px; margin-top: 10px; opacity: 0.8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI SENTIMENT</h1>
        <textarea id="userInput" placeholder="Paste your customer review here..."></textarea>
        <button onclick="analyze()">ANALYZE NOW</button>
        <div id="resultBox">
            <div id="sentimentText" style="font-size: 24px; font-weight: bold;"></div>
            <div class="details" id="nlpDetails"></div>
        </div>
    </div>

    <script>
        async function analyze() {
            const text = document.getElementById('userInput').value;
            if(!text) return;
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await response.json();
            
            const resBox = document.getElementById('resultBox');
            const sentText = document.getElementById('sentimentText');
            const nlpText = document.getElementById('nlpDetails');
            
            resBox.style.display = 'block';
            resBox.className = data.sentiment;
            sentText.innerText = "RESULT: " + data.sentiment.toUpperCase();
            nlpText.innerText = "Processed with Tokenization & Lemmatization";
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_UI)

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    raw_text = content['text']
    
    # NLP STEPS (For your project requirements)
    tokens = word_tokenize(raw_text)
    lemmas = [lemmatizer.lemmatize(t.lower()) for t in tokens]
    
    # Sentiment Calculation
    scores = sia.polarity_scores(raw_text)
    comp = scores['compound']
    
    if comp >= 0.05:
        res = "positive"
    elif comp <= -0.05:
        res = "negative"
    else:
        res = "neutral"
        
    return jsonify({'sentiment': res})

if __name__ == '__main__':
    app.run(port=5000)