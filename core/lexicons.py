# Institutional Teaching Aspects & Lexicons (Section 3.5.2 & 3.6.2)

ASPECTS = [
    "Teaching Clarity",
    "Course Organisation",
    "Assessment and Grading",
    "Lecturer Punctuality",
    "Availability",
    "Communication",
    "Student Interaction",
    "Use of Teaching Materials"
]

ASPECT_KEYWORDS = {
    "Teaching Clarity": {
        "positive": [
            "explains clearly", "easy to understand", "breaks down", "simplifies",
            "good examples", "makes sense", "easy to follow", "clearly",
            "articulate", "comprehensible", "digestible", "crystal clear",
            "clear explanation", "clarity", "great explanations", "well explained",
            "explain difficult concepts simple", "explanation sharp", "sabi book",
            "sabi", "sabi die", "enter head", "break down complex", "tear lecture",
            "wella", "dey enter head"
        ],
        "negative": [
            "does not explain", "doesn't explain", "difficult to understand", "hard to understand",
            "complicates", "rushes through", "confusing", "hard to follow",
            "disorganized explanation", "unclear", "speaks too fast", "vague", "jumps between topics",
            "poor explanation", "cannot understand", "hard to comprehend",
            "explanation no clear", "nobody dey grab", "talk to himself",
            "mumu", "mumu man", "no sabi", "talk to himself", "nobody dey grab", "no head",
            "scatter brain", "speak in tongues", "sleep for class", "read slides",
            "nothing dey enter head", "olodo", "clown", "waste", "idiot",
            "foolish", "stupid", "idiotic", "nonsense", "rubbish"
        ],
        "neutral": ["standard explanation", "textbook explanation", "moderate", "adequate", "syllabus coverage"]
    },
    "Course Organisation": {
        "positive": [
            "exceptionally organized", "well organized", "follows the syllabus", "schedule",
            "clear outline", "milestones", "logical progression", "pace is managed", "structured",
            "course is organized", "well structured", "organized lectures",
            "course outline well organized", "no rush at all"
        ],
        "negative": [
            "poorly organized", "no clear syllabus", "disorganized", "rushed in the final weeks",
            "lacks structure", "disjointed", "shifted erratically", "unorganized", "chaotic course",
            "no proper syllabus", "topics jumping around", "course outline completely disorganized"
        ],
        "neutral": ["standard syllabus", "assigned curriculum", "course outline", "departmental calendar"]
    },
    "Assessment and Grading": {
        "positive": [
            "fair grading", "grades fairly", "detailed feedback", "exams match",
            "returns quickly", "clear rubrics", "transparent grading", "constructive feedback",
            "objective", "marking is fair", "grading is fair", "fair marking", "fair tests",
            "marks are fair", "helpful feedback", "ca test was fair", "results came out quick",
            "mark well", "if you sabi you go pass", "just marking", "no wahala"
        ],
        "negative": [
            "unfair grading", "arbitrary", "inconsistent", "never taught",
            "takes weeks", "no rubric", "random marks", "harsh grading",
            "scripts never returned", "vague feedback", "grading was unfair", "grading is unfair",
            "unfair marking", "harsh marking", "harsh", "unfair exams", "unreasonable exams", "harshly",
            "grading harsh", "person write well still fail", "mark wickedly", "fail for nothing",
            "mark anyhow", "mark randomly", "fail people like competition",
            "likes money", "greedy", "money", "buy textbook", "force us to buy"
        ],
        "neutral": ["standard grading", "normal scale", "two tests and one exam", "administered as scheduled"]
    },
    "Lecturer Punctuality": {
        "positive": [
            "punctual", "always on time", "arrives early", "regular attendance",
            "never misses", "utilizes full period", "prompt arrival", "starts on time",
            "very punctual", "consistently on time", "punctuality",
            "always come class early"
        ],
        "negative": [
            "habitually late", "always late", "comes late", "cancels lectures",
            "missed classes", "wastes class time", "poor punctuality", "behind schedule",
            "not punctual", "arrives late", "cancels class",
            "frequently miss lectures"
        ],
        "neutral": ["arrives on time", "grace period", "timing was maintained", "reasonable attendance"]
    },
    "Availability": {
        "positive": [
            "available during office", "office hours", "approachable", "makes time",
            "willing to assist", "readily accessible", "door is open", "mentors students",
            "always available", "accessible", "easy to reach", "helpful outside class",
            "dey reply emails", "very approachable", "sharp sharp", "no dulling",
            "dey make time", "easy to reach", "no wahala"
        ],
        "negative": [
            "impossible to reach", "never available", "doors locked", "dismissive",
            "unreachable", "impatient", "cannot reach", "cancels office",
            "not available", "hard to reach", "unavailable",
            "office door always locked", "impossible to see him for office",
            "no dey answer", "very unapproachable", "no dey reply", "run from students",
            "impossible to see", "shun students", "office door locked"
        ],
        "neutral": ["office hours by appointment", "consultation hours", "standard university hours"]
    },
    "Communication": {
        "positive": [
            "responds promptly", "replies quickly", "clear announcements",
            "proactive communication", "courteous communication", "well informed", "informative",
            "communicates well", "responsive to emails", "prompt replies"
        ],
        "negative": [
            "fails to respond", "ignores emails", "unanswered", "confusing announcements",
            "uninformed", "last-minute", "poor communication", "unreliable",
            "does not reply", "slow communication"
        ],
        "neutral": ["announcements via rep", "class representative", "notice board", "portal notices"]
    },
    "Student Interaction": {
        "positive": [
            "interactive", "engaging", "encourages questions", "active participation",
            "listens attentively", "respectful", "welcoming debates", "dynamic class",
            "good interaction", "classroom interaction", "treats students with respect"
        ],
        "negative": [
            "intimidates students", "one-way monologue", "dull", "discourages questions",
            "dismisses questions", "hostile", "unengaging", "ridicules students",
            "no interaction", "monotonous", "ignores questions",
            "personal issues with students", "likes helpless", "female students"
        ],
        "neutral": ["standard lecture", "some questions", "large lecture hall", "instructional"]
    },
    "Use of Teaching Materials": {
        "positive": [
            "detailed slides", "comprehensive slides", "shares notes", "great materials",
            "multimedia", "projector", "handouts", "up-to-date", "curated readings",
            "slides are helpful", "good lecture notes", "useful slides"
        ],
        "negative": [
            "no slides", "refuses to share notes", "outdated materials",
            "blurry slides", "poor materials", "typos and errors", "zero supplementary",
            "bad slides", "outdated slides", "lacks materials",
            "force us to buy", "if you no buy textbook"
        ],
        "neutral": ["standard slides", "textbook recommended", "faculty materials", "bullet points"]
    }
}

NEGATION_MARKERS = {
    "not", "no", "never", "n't", "none", "nobody", "nothing", "neither",
    "nowhere", "hardly", "scarcely", "barely", "impossible", "unanswered",
    "cancels", "doesn't", "don't", "won't", "can't", "couldn't",
    "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "didn't", "without", "rarely",
    "no be", "no get", "no dey", "no sabi"
}

