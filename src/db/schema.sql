PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- ============================================================
-- 1. USERS
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT,

    -- Store a password hash only. Never store the plain password.
    password_hash TEXT NOT NULL,

    age INTEGER,
    avg_reading_lvl TEXT,

    updated_at TEXT,

    CHECK (length(password_hash) > 0)
);

-- Auto-update updated_at only when user-editable fields change.
-- The UPDATE OF clause and WHEN condition prevent repeated recursive updates.
CREATE TRIGGER IF NOT EXISTS trg_users_updated_at
AFTER UPDATE OF
    username,
    display_name,
    password_hash,
    age,
    avg_reading_lvl
ON users
FOR EACH ROW
WHEN NEW.updated_at IS OLD.updated_at
BEGIN
    UPDATE users
    SET updated_at = CURRENT_TIMESTAMP
    WHERE user_id = OLD.user_id;
END;

-- ============================================================
-- 2. VOCABULARY TERMS
-- ============================================================
-- Stores beginner, intermediate, advanced, and unknown/discovered words.
-- Unknown words can be inserted here with level = 'Unknown'.

CREATE TABLE IF NOT EXISTS vocabulary_terms (
    term_id INTEGER PRIMARY KEY AUTOINCREMENT,

    word TEXT NOT NULL,

    level TEXT NOT NULL DEFAULT 'Unknown',

    source TEXT,
    notes TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (level IN ('Beginner', 'Intermediate', 'Advanced', 'Unknown')),

    UNIQUE (word, level)
);


-- ============================================================
-- 3. BOOKS
-- ============================================================

CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,

    title TEXT NOT NULL,
    author TEXT,

    source_file TEXT,
    cleaned_file TEXT,
    source_hash TEXT UNIQUE,

    language TEXT NOT NULL DEFAULT 'en',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT
);

-- Auto-update updated_at only when book metadata fields change.
-- The UPDATE OF clause and WHEN condition prevent repeated recursive updates.
CREATE TRIGGER IF NOT EXISTS trg_books_updated_at
AFTER UPDATE OF
    title,
    author,
    source_file,
    cleaned_file,
    source_hash,
    language
ON books
FOR EACH ROW
WHEN NEW.updated_at IS OLD.updated_at
BEGIN
    UPDATE books
    SET updated_at = CURRENT_TIMESTAMP
    WHERE book_id = OLD.book_id;
END;

-- ============================================================
-- 4. BOOK SEGMENTS
-- ============================================================
-- Stores chapters, pages, sections, or fixed-size chunks from a book.

CREATE TABLE IF NOT EXISTS book_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,

    book_id INTEGER NOT NULL,

    segment_type TEXT NOT NULL DEFAULT 'chunk',
    segment_index INTEGER NOT NULL,

    heading TEXT,
    text TEXT NOT NULL,

    char_start INTEGER,
    char_end INTEGER,

    word_count INTEGER,
    token_count INTEGER,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE,

    CHECK (segment_type IN ('book', 'chapter', 'page', 'section', 'chunk')),
    CHECK (word_count IS NULL OR word_count >= 0),
    CHECK (token_count IS NULL OR token_count >= 0),

    UNIQUE (book_id, segment_type, segment_index)
);

-- ============================================================
-- 5. USER SOURCES
-- ============================================================
-- Stores text that comes from the user:
-- uploaded book, pasted text, writing sample, quiz response, etc.

CREATE TABLE IF NOT EXISTS user_sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER NOT NULL,

    title TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'uploaded_text',

    source_file TEXT,

    raw_text TEXT,
    cleaned_text TEXT,
    source_hash TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,

    CHECK (source_type IN (
        'uploaded_book',
        'uploaded_text',
        'writing_sample',
        'quiz_response',
        'reading_assignment'
    ))
);

-- ============================================================
-- 6. USER SOURCE SEGMENTS
-- ============================================================
-- Stores chunks from user-uploaded or user-written sources.

CREATE TABLE IF NOT EXISTS user_source_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,

    source_id INTEGER NOT NULL,

    segment_index INTEGER NOT NULL,
    text TEXT NOT NULL,

    word_count INTEGER,
    token_count INTEGER,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (source_id) REFERENCES user_sources(source_id) ON DELETE CASCADE,

    CHECK (word_count IS NULL OR word_count >= 0),
    CHECK (token_count IS NULL OR token_count >= 0),

    UNIQUE (source_id, segment_index)
);

-- ============================================================
-- 7. GENERATED STORIES
-- ============================================================
-- Stores custom stories created for a user.
-- This table also stores final story-level analysis values so that
-- a separate story_level_results table is not needed.

CREATE TABLE IF NOT EXISTS generated_stories (
    story_id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER NOT NULL,

    title TEXT NOT NULL,
    story_text TEXT NOT NULL,

    target_level TEXT NOT NULL,
    target_grade_band TEXT,

    prompt_used TEXT,
    model_name TEXT,

    requested_word_count INTEGER,
    actual_word_count INTEGER,

    practice_words_json TEXT,
    avoided_words_json TEXT,

    beginner_ratio REAL,
    intermediate_ratio REAL,
    advanced_ratio REAL,
    unknown_word_ratio REAL,
    academic_word_ratio REAL,

    analyzed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,

    CHECK (target_level IN ('Beginner', 'Intermediate', 'Advanced', 'Unknown')),

    CHECK (requested_word_count IS NULL OR requested_word_count >= 0),
    CHECK (actual_word_count IS NULL OR actual_word_count >= 0),

    CHECK (beginner_ratio IS NULL OR beginner_ratio BETWEEN 0.0 AND 1.0),
    CHECK (intermediate_ratio IS NULL OR intermediate_ratio BETWEEN 0.0 AND 1.0),
    CHECK (advanced_ratio IS NULL OR advanced_ratio BETWEEN 0.0 AND 1.0),
    CHECK (unknown_word_ratio IS NULL OR unknown_word_ratio BETWEEN 0.0 AND 1.0),
    CHECK (academic_word_ratio IS NULL OR academic_word_ratio BETWEEN 0.0 AND 1.0)
);

-- ============================================================
-- 8. ANALYSIS RUNS
-- ============================================================
-- Tracks every time a script analyzes a book, user source,
-- generated story, or recalculates a user reading profile.

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_target_type TEXT NOT NULL,

    user_id INTEGER,
    book_id INTEGER,
    source_id INTEGER,
    story_id INTEGER,

    run_name TEXT,

    script_name TEXT,
    script_version TEXT,

    model_name TEXT,
    tokenizer_name TEXT,

    chunk_size INTEGER,
    parameters_json TEXT,

    status TEXT NOT NULL DEFAULT 'started',

    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,

    notes TEXT,

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES user_sources(source_id) ON DELETE CASCADE,
    FOREIGN KEY (story_id) REFERENCES generated_stories(story_id) ON DELETE CASCADE,

    CHECK (run_target_type IN (
        'book',
        'user_source',
        'generated_story',
        'user_profile'
    )),

    CHECK (status IN ('started', 'completed', 'failed')),

    CHECK (chunk_size IS NULL OR chunk_size > 0),

    CHECK (
        (
            run_target_type = 'book'
            AND book_id IS NOT NULL
            AND source_id IS NULL
            AND story_id IS NULL
        )
        OR
        (
            run_target_type = 'user_source'
            AND user_id IS NOT NULL
            AND source_id IS NOT NULL
            AND book_id IS NULL
            AND story_id IS NULL
        )
        OR
        (
            run_target_type = 'generated_story'
            AND user_id IS NOT NULL
            AND story_id IS NOT NULL
            AND book_id IS NULL
            AND source_id IS NULL
        )
        OR
        (
            run_target_type = 'user_profile'
            AND user_id IS NOT NULL
            AND book_id IS NULL
            AND source_id IS NULL
            AND story_id IS NULL
        )
    )
);

-- ============================================================
-- 9. SEGMENT WORD COUNTS
-- ============================================================
-- Stores word frequencies from book segments, user source segments,
-- or whole generated stories.
--
-- For generated stories, use:
-- segment_target_type = 'generated_story'
-- story_id = the story
-- segment_index = 0
--
-- That avoids needing a generated_story_segments table.

CREATE TABLE IF NOT EXISTS segment_word_counts (
    word_count_id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id INTEGER NOT NULL,

    segment_target_type TEXT NOT NULL,

    book_segment_id INTEGER,
    user_segment_id INTEGER,
    story_id INTEGER,

    segment_index INTEGER NOT NULL DEFAULT 0,

    term_id INTEGER,

    word TEXT NOT NULL,
    lemma TEXT NOT NULL,
    pos TEXT NOT NULL DEFAULT 'unknown',

    count INTEGER NOT NULL,

    is_oov INTEGER NOT NULL DEFAULT 1,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (book_segment_id) REFERENCES book_segments(segment_id) ON DELETE CASCADE,
    FOREIGN KEY (user_segment_id) REFERENCES user_source_segments(segment_id) ON DELETE CASCADE,
    FOREIGN KEY (story_id) REFERENCES generated_stories(story_id) ON DELETE CASCADE,
    FOREIGN KEY (term_id) REFERENCES vocabulary_terms(term_id) ON DELETE SET NULL,

    CHECK (segment_target_type IN (
        'book',
        'user_source',
        'generated_story'
    )),

    CHECK (count >= 0),
    CHECK (is_oov IN (0, 1)),

    CHECK (
        (
            segment_target_type = 'book'
            AND book_segment_id IS NOT NULL
            AND user_segment_id IS NULL
            AND story_id IS NULL
        )
        OR
        (
            segment_target_type = 'user_source'
            AND user_segment_id IS NOT NULL
            AND book_segment_id IS NULL
            AND story_id IS NULL
        )
        OR
        (
            segment_target_type = 'generated_story'
            AND story_id IS NOT NULL
            AND book_segment_id IS NULL
            AND user_segment_id IS NULL
        )
    )
);

-- ============================================================
-- 10. USER WORD KNOWLEDGE
-- ============================================================
-- Stores the system's estimate of what each user knows.

CREATE TABLE IF NOT EXISTS user_word_knowledge (
    user_id INTEGER NOT NULL,
    term_id INTEGER NOT NULL,

    knowledge_status TEXT NOT NULL DEFAULT 'unknown',
    confidence REAL NOT NULL DEFAULT 0.0,

    times_seen INTEGER NOT NULL DEFAULT 0,
    times_used INTEGER NOT NULL DEFAULT 0,
    times_correct INTEGER NOT NULL DEFAULT 0,
    times_incorrect INTEGER NOT NULL DEFAULT 0,

    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,

    -- FK to the analysis run that last changed this row.
    -- Useful for debugging why the model assigned a given status.
    last_updated_by_run_id INTEGER,

    evidence_json TEXT,

    PRIMARY KEY (user_id, term_id),

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (term_id) REFERENCES vocabulary_terms(term_id) ON DELETE CASCADE,
    FOREIGN KEY (last_updated_by_run_id) REFERENCES analysis_runs(run_id) ON DELETE SET NULL,

    CHECK (knowledge_status IN (
        'unknown',
        'learning',
        'likely_known',
        'known',
        'struggling'
    )),

    CHECK (confidence BETWEEN 0.0 AND 1.0),

    CHECK (times_seen >= 0),
    CHECK (times_used >= 0),
    CHECK (times_correct >= 0),
    CHECK (times_incorrect >= 0)
);

-- ============================================================
-- 11. USER READING PROFILES
-- ============================================================
-- Stores snapshots of a user's estimated reading level over time.
-- Your Python script should set old rows to is_current = 0
-- before inserting a new current profile.

CREATE TABLE IF NOT EXISTS user_reading_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER NOT NULL,
    run_id INTEGER,

    estimated_level TEXT NOT NULL DEFAULT 'Unknown',
    estimated_grade_band TEXT,

    known_beginner_words INTEGER NOT NULL DEFAULT 0,
    known_intermediate_words INTEGER NOT NULL DEFAULT 0,
    known_advanced_words INTEGER NOT NULL DEFAULT 0,
    known_unknown_level_words INTEGER NOT NULL DEFAULT 0,

    total_known_words INTEGER NOT NULL DEFAULT 0,
    academic_words_known INTEGER NOT NULL DEFAULT 0,

    beginner_mastery REAL NOT NULL DEFAULT 0.0,
    intermediate_mastery REAL NOT NULL DEFAULT 0.0,
    advanced_mastery REAL NOT NULL DEFAULT 0.0,

    confidence REAL NOT NULL DEFAULT 0.0,

    profile_json TEXT,

    is_current INTEGER NOT NULL DEFAULT 1,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE SET NULL,

    CHECK (estimated_level IN (
        'Beginner',
        'Intermediate',
        'Advanced',
        'Unknown'
    )),

    CHECK (known_beginner_words >= 0),
    CHECK (known_intermediate_words >= 0),
    CHECK (known_advanced_words >= 0),
    CHECK (known_unknown_level_words >= 0),
    CHECK (total_known_words >= 0),
    CHECK (academic_words_known >= 0),

    CHECK (beginner_mastery BETWEEN 0.0 AND 1.0),
    CHECK (intermediate_mastery BETWEEN 0.0 AND 1.0),
    CHECK (advanced_mastery BETWEEN 0.0 AND 1.0),

    CHECK (confidence BETWEEN 0.0 AND 1.0),
    CHECK (is_current IN (0, 1))
);

-- ============================================================
-- 12. BOOK RECOMMENDATIONS
-- ============================================================
-- Stores books recommended to a user and the reason/score.
-- is_current mirrors the pattern in user_reading_profiles:
-- set old rows to 0 when a new profile generates fresh recommendations.

CREATE TABLE IF NOT EXISTS book_recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER NOT NULL,
    profile_id INTEGER,
    book_id INTEGER NOT NULL,

    recommended_level TEXT NOT NULL DEFAULT 'Unknown',

    match_score REAL NOT NULL,

    known_word_ratio REAL,
    unknown_word_ratio REAL,
    advanced_word_ratio REAL,
    academic_word_ratio REAL,

    reason TEXT,

    -- Set to 0 when the user's profile is updated and new recommendations
    -- are generated, so stale rows are never surfaced by accident.
    is_current INTEGER NOT NULL DEFAULT 1,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES user_reading_profiles(profile_id) ON DELETE SET NULL,
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE,

    CHECK (recommended_level IN (
        'Beginner',
        'Intermediate',
        'Advanced',
        'Unknown'
    )),

    CHECK (match_score BETWEEN 0.0 AND 1.0),

    CHECK (known_word_ratio IS NULL OR known_word_ratio BETWEEN 0.0 AND 1.0),
    CHECK (unknown_word_ratio IS NULL OR unknown_word_ratio BETWEEN 0.0 AND 1.0),
    CHECK (advanced_word_ratio IS NULL OR advanced_word_ratio BETWEEN 0.0 AND 1.0),
    CHECK (academic_word_ratio IS NULL OR academic_word_ratio BETWEEN 0.0 AND 1.0),

    CHECK (is_current IN (0, 1))
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_users_username
ON users(username);

CREATE INDEX IF NOT EXISTS idx_vocab_word
ON vocabulary_terms(word);

CREATE INDEX IF NOT EXISTS idx_vocab_level
ON vocabulary_terms(level);

CREATE INDEX IF NOT EXISTS idx_books_title
ON books(title);

CREATE INDEX IF NOT EXISTS idx_books_author
ON books(author);

CREATE INDEX IF NOT EXISTS idx_book_segments_book
ON book_segments(book_id);

CREATE INDEX IF NOT EXISTS idx_book_segments_type_index
ON book_segments(book_id, segment_type, segment_index);

CREATE INDEX IF NOT EXISTS idx_user_sources_user
ON user_sources(user_id);

CREATE INDEX IF NOT EXISTS idx_user_sources_type
ON user_sources(source_type);

CREATE INDEX IF NOT EXISTS idx_user_source_segments_source
ON user_source_segments(source_id);

CREATE INDEX IF NOT EXISTS idx_generated_stories_user
ON generated_stories(user_id);

CREATE INDEX IF NOT EXISTS idx_generated_stories_level
ON generated_stories(target_level);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_target
ON analysis_runs(run_target_type);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_user
ON analysis_runs(user_id);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_book
ON analysis_runs(book_id);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_source
ON analysis_runs(source_id);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_story
ON analysis_runs(story_id);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_status
ON analysis_runs(status);

CREATE INDEX IF NOT EXISTS idx_word_counts_run
ON segment_word_counts(run_id);

CREATE INDEX IF NOT EXISTS idx_word_counts_target
ON segment_word_counts(segment_target_type);

CREATE INDEX IF NOT EXISTS idx_word_counts_term
ON segment_word_counts(term_id);

CREATE INDEX IF NOT EXISTS idx_word_counts_lemma
ON segment_word_counts(lemma);

CREATE INDEX IF NOT EXISTS idx_word_counts_book_segment
ON segment_word_counts(book_segment_id);

CREATE INDEX IF NOT EXISTS idx_word_counts_user_segment
ON segment_word_counts(user_segment_id);

CREATE INDEX IF NOT EXISTS idx_word_counts_story
ON segment_word_counts(story_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_word_counts_book_segment
ON segment_word_counts(run_id, book_segment_id, lemma, pos)
WHERE book_segment_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_word_counts_user_segment
ON segment_word_counts(run_id, user_segment_id, lemma, pos)
WHERE user_segment_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_word_counts_generated_story
ON segment_word_counts(run_id, story_id, segment_index, lemma, pos)
WHERE story_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_user_word_knowledge_user
ON user_word_knowledge(user_id);

CREATE INDEX IF NOT EXISTS idx_user_word_knowledge_term
ON user_word_knowledge(term_id);

CREATE INDEX IF NOT EXISTS idx_user_word_knowledge_status
ON user_word_knowledge(knowledge_status);

CREATE INDEX IF NOT EXISTS idx_user_word_knowledge_confidence
ON user_word_knowledge(confidence);

-- Index to support tracing which run last updated a knowledge row
CREATE INDEX IF NOT EXISTS idx_user_word_knowledge_run
ON user_word_knowledge(last_updated_by_run_id);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user
ON user_reading_profiles(user_id);

CREATE INDEX IF NOT EXISTS idx_user_profiles_current
ON user_reading_profiles(user_id, is_current);

-- Enforce only one current reading profile per user.
CREATE UNIQUE INDEX IF NOT EXISTS uq_user_one_current_profile
ON user_reading_profiles(user_id)
WHERE is_current = 1;

CREATE INDEX IF NOT EXISTS idx_user_profiles_level
ON user_reading_profiles(estimated_level);

CREATE INDEX IF NOT EXISTS idx_book_recommendations_user
ON book_recommendations(user_id);

CREATE INDEX IF NOT EXISTS idx_book_recommendations_book
ON book_recommendations(book_id);

CREATE INDEX IF NOT EXISTS idx_book_recommendations_score
ON book_recommendations(match_score);

-- Index to make filtering current recommendations fast
CREATE INDEX IF NOT EXISTS idx_book_recommendations_current
ON book_recommendations(user_id, is_current);

-- Index to make current recommendations sorted by score fast
CREATE INDEX IF NOT EXISTS idx_book_recommendations_current_score
ON book_recommendations(user_id, is_current, match_score);

COMMIT;
