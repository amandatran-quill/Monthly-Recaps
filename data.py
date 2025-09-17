"""
Pipeline for Top Tag Chat Deep Dive (Custom Folder Names)
Author: [Your Name]
"""

import os
import glob
import pandas as pd
import re
from collections import Counter
import nltk


def ensure_nltk_resources():
    """Ensure required NLTK resources are present; download if missing."""
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for res in resources:
        try:
            # attempt to locate the resource; for tokenizers this path works
            if res == 'stopwords':
                nltk.data.find('corpora/stopwords')
            else:
                # tokenizers/punkt or tokenizers/punkt_tab
                nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                # best-effort; continue
                pass


ensure_nltk_resources()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Filtering lists ---
TAGS_TO_REMOVE = [
    "Free", "Not tagged", "Conversation Rating - High Score", "District Premium",
    "Teacher Premium", "School Premium", "Conversation Rating - Low Score"
]
CUSTOM_STOPWORDS = set([
    # Company/agent/system words
    'quill', 'support', 'team', 'operator', 'article', 'inserter', 
    'diagnostic', 'pack', 'activity', 'activities', 'test', 'class', 
    'report', 'practice', 'lesson',
    'assign', 'tab', 'menu', 'dashboard', 'pre', 'code',
    'survey', 'feature', 'invite', 'update',

    # Company/agent names
    'nikki', 'amanda', 'nattalie', 'shannon', 'erika','alex','charlie',

    # Signatures and contact lines
    'best', 'regards', 'thank', 'thanks', 'sincerely', 'please', 
    'thank you', 'welcome', 'appreciate', 'help', 'let', 'know',
    'pleasure', 'contact', 'support', 'assist', 'further', 'clarify', 'additional', 
    'follow', 'question', 'questions', 'concerns', 'grettings',

    # Greetings and fillers
    'hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening',
    'hope', 'day', 'back', 'next',

    # System/meta
    'conversation', 'started', 'exported', 'reply', 'replies', 'email', 'recipient',
    'message', 'system', 'notice', 'confidentiality', 'transmitted', 'error',
    'reading', 'distribution', 'copying', 'strictly', 'prohibited', 'received', 
    'immediately', 'delete', 'copies', 'backups', 'sent', 'attached', 'include',
    'time', 'date', 'gmt', 'ed', 'edt', 'est', 'am', 'pm', 'today', 'yesterday', 'tomorrow',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',

    # Miscellaneous/URLs/emails/domains
    'mailto', 'com', 'org', 'us', 'kyschools', 'net', 'helpcenter', 'supportquill',
    'quillorg', 'classcode', 'google', 'clever', 'schools', 'survey', 'questionfeedback',

    # Punctuation and common formatting artifacts
    '—', '–', '“', '”', '…', '(', ')', '[', ']', '{', '}', '’', '‘'
])

# --- Helper functions ---
def is_unwanted_token(token):
    if re.match(r'\d{1,2}:\d{2}', token): return True  # Time
    if re.match(r'20\d{2}', token): return True        # Year
    if '@' in token or 'www.' in token: return True    # Emails, websites
    if token in CUSTOM_STOPWORDS: return True
    return False

def clean_text_advanced(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Prefer NLTK tokenization, but fall back to a simple regex tokenizer
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = re.findall(r"\b[a-z]{3,}\b", text)
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set()
    tokens = [w for w in tokens if w not in stop_words and not is_unwanted_token(w) and len(w) > 2]
    return tokens

def extract_user_messages_from_log(text):
    """Return all user/customer messages from a raw chat log (Quill support pattern)."""
    messages = []
    for line in text.split('\n'):
        match = re.match(r'^\d{1,2}:\d{2}\s*[AP]M\s*\|\s*([^:]+?):\s*(.*)$', line)
        if match:
            sender, message = match.group(1).strip(), match.group(2).strip()
            if not re.search(r'Quill|Nattalie|Nikki|Amanda|Operator|The Quill Team', sender, re.IGNORECASE):
                messages.append(message)
    return messages

# --- Folder mapping ---
# Map each tag to its folder (change this if folder names change)
TAG_TO_FOLDER = {
    "My Account": "My Account Aug–Sep16",
    "Manage Activities": "Manage Activities Aug–Sep16",
    "Assign PR": "Assign PR Aug–Sep16",
    "Manage Classes": "Manage Classes Aug–Sep16"
}

# --- Load main summary, determine top 4/5 tags ---
# Discover CSV (allow it to live in the Aug-Sep 2025 subfolder)
possible_csvs = glob.glob(os.path.join(os.getcwd(), '**', 'custom_chart_2025-08-20_2025-09-16.csv'), recursive=True)
if not possible_csvs:
    raise FileNotFoundError('Could not find custom_chart_2025-08-20_2025-09-16.csv in the workspace')
main_csv = possible_csvs[0]
df = pd.read_csv(main_csv)

# The CSV header we expect in older exports is 'Conversation tag' but some exports
# have variations (e.g. line breaks in header). Normalize column names and try
# to find the best match.
def find_conversation_tag_column(columns):
    norm = {c: re.sub(r'\s+', ' ', c).strip().lower() for c in columns}
    for c, nc in norm.items():
        if 'conversation' in nc and 'tag' in nc:
            return c
    # fallback: any column containing 'tag' or 'conversation'
    for c, nc in norm.items():
        if 'tag' in nc or 'conversation' in nc:
            return c
    return None

conv_col = find_conversation_tag_column(df.columns.tolist())
if conv_col is None:
    raise KeyError(f"Could not find a 'Conversation tag' column in {main_csv}. Columns: {df.columns.tolist()}")

# The export CSV includes metadata rows above a simple 2-column table. We'll
# search the file for the header row "Conversation tag","New conversations"
# and parse the following lines as the table.
def parse_export_two_column_table(csv_path):
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    # find index of the exact header row (robust to quotes/spacing)
    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip().lower().replace('"', '').replace("'", '').strip()
        # Expect something like: conversation tag,new conversations
        if s.startswith('conversation tag') and 'new conversations' in s:
            header_idx = i
            break
    if header_idx is None:
        # fallback: try using detected conv_col as a single-column CSV
        # In that case, return the original dataframe filtered by tags-to-remove
        df_clean = df[~df[conv_col].isin(TAGS_TO_REMOVE)].copy()
        return df_clean.rename(columns={conv_col: 'Conversation tag'})

    # Build a small CSV from header_idx and onwards
    table_lines = lines[header_idx:]
    from io import StringIO
    table_buf = StringIO(''.join(table_lines))
    # Let pandas parse the header row. We'll detect which columns correspond to
    # conversation tag and counts in a flexible way.
    table_df = pd.read_csv(table_buf, header=0)
    # Find the columns for conversation tag and counts
    col_map = {c: re.sub(r'\s+', ' ', c).strip().lower() for c in table_df.columns.tolist()}
    tag_col = None
    count_col = None
    for c, nc in col_map.items():
        if 'conversation' in nc and 'tag' in nc:
            tag_col = c
        if 'new' in nc and ('conversations' in nc or 'conversation' in nc or 'count' in nc):
            count_col = c
    # Fallback heuristics
    if tag_col is None:
        for c, nc in col_map.items():
            if 'tag' in nc or 'conversation' in nc:
                tag_col = c
                break
    if count_col is None:
        for c, nc in col_map.items():
            if re.search(r'\d|new|count|conversations', nc):
                count_col = c
                break
    if tag_col is None or count_col is None:
        # Give up and return an empty dataframe with expected columns
        return pd.DataFrame(columns=['Conversation tag', 'New conversations'])
    # coerce counts to int where possible
    table_df[count_col] = pd.to_numeric(table_df[count_col], errors='coerce').fillna(0).astype(int)
    table_df = table_df.rename(columns={tag_col: 'Conversation tag', count_col: 'New conversations'})
    # filter unwanted tags
    table_df = table_df[~table_df['Conversation tag'].isin(TAGS_TO_REMOVE)]
    # sort descending
    table_df = table_df.sort_values('New conversations', ascending=False).reset_index(drop=True)
    return table_df

df_top = parse_export_two_column_table(main_csv)
df_top.to_csv('top_tags_summary.csv', index=False)

# --- Process each tag and its folder ---
# NOTE: deep-dive processing is independent from the top-tags CSV step.
# We will attempt to read .txt chat logs from either a directory named in
# `TAG_TO_FOLDER` or from a zip file with the same name. This allows you to
# keep zipped exports in the project folder.
def list_txt_files_in_folder_or_zip(folder_name):
    """Return list of absolute paths to .txt files inside a folder or inside a zip.
    If a zip is found, extract in-memory listing and return paths in the format 'zip://<zipfile>::<internalpath>'"""
    results = []
    # Allow hyphen/en-dash variants and check both relative and inside 'Aug-Sep 2025'
    candidates = [folder_name, folder_name.replace('–', '-'), os.path.join('Aug-Sep 2025', folder_name), os.path.join('Aug-Sep 2025', folder_name.replace('–','-'))]
    for cand in candidates:
        cand_path = os.path.join(os.getcwd(), cand)
        # directory case
        if os.path.isdir(cand_path):
            for p in glob.glob(os.path.join(cand_path, '*.txt')):
                results.append(p)
            if results:
                return results
        # zip case
        zip_path = cand_path + '.zip'
        if os.path.isfile(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                for zi in z.namelist():
                    if zi.lower().endswith('.txt'):
                        # use a pseudo-path that includes the zipfile so we can open it later
                        results.append(f'zip://{zip_path}::{zi}')
            if results:
                return results
    return results

for tag_key, folder in TAG_TO_FOLDER.items():
    tag_csv_name_part = tag_key.replace(" ", "_")
    # We no longer require the tag to be present in the CSV to run the deep dive.
    tag_name = tag_key
    print(f"\nProcessing Tag: {tag_name} | Folder: {folder}")
    txt_files = list_txt_files_in_folder_or_zip(folder)
    if not txt_files:
        print(f"  No .txt chat logs found for '{folder}', looked for folder/zip variants, skipping.")
        continue

    chat_rows = []
    for fname in txt_files:
        if fname.startswith('zip://'):
            # format: zip://<zip_path>::<internal>
            _, rest = fname.split('zip://', 1)
            zip_path, internal = rest.split('::', 1)
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(internal) as f:
                    raw_bytes = f.read()
                    try:
                        raw_text = raw_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        raw_text = raw_bytes.decode('latin-1')
        else:
            # Try utf-8, fallback to latin-1 for files with different encodings
            tried = False
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    tried = True
            except UnicodeDecodeError:
                with open(fname, 'r', encoding='latin-1') as f:
                    raw_text = f.read()
        messages = extract_user_messages_from_log(raw_text)
        for m in messages:
            chat_rows.append({'user_message': m})

    df_chat = pd.DataFrame(chat_rows)
    if df_chat.empty:
        print("  No user messages extracted, skipping.")
        continue

    df_chat['is_question'] = df_chat['user_message'].apply(lambda x: x.strip().endswith('?'))
    df_chat['tokens'] = df_chat['user_message'].apply(clean_text_advanced)

    # Deep dive: Save user questions and top keywords
    questions = df_chat[df_chat['is_question']]['user_message'].tolist()
    questions_csv = f"{tag_csv_name_part}_user_questions.csv"
    pd.DataFrame({'user_message': questions}).to_csv(questions_csv, index=False)

    all_tokens = [token for tokens in df_chat[df_chat['is_question']]['tokens'] for token in tokens]
    keywords_df = pd.DataFrame(Counter(all_tokens).most_common(20), columns=['keyword', 'frequency'])
    keywords_csv = f"{tag_csv_name_part}_question_keywords.csv"
    keywords_df.to_csv(keywords_csv, index=False)

    print(f"  Saved: {questions_csv}, {keywords_csv}")

print("\nAll folders processed. See CSV files for your deep dive results.")
