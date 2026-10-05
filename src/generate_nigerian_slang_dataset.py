import random
import pandas as pd
import json
random.seed(42)

# Nigerian slang and informal expressions for each aspect
nigerian_slang_templates = {
    "Teaching Clarity": {
        "positive": [
            "This guy sabi book die, explanations dey enter head well well",
            "E dey break down complex things like say na jollof rice",
            "Every lecture dey clear, no wahala at all",
            "Man sabi explain tire, even dullard go understand",
            "E dey use better examples wey dey relate to our everyday life",
            "Explains things way better than those old lecturers wey dey confuse person",
            "E dey make the hard topics look like child's play",
            "If you no understand after this guy's class, then na you be the problem",
            "E dey tear lecture wella, everything dey crystal clear",
            "The way e dey explain dey give joy, you go just understand everything"
        ],
        "negative": [
            "This mumu man no sabi explain anything at all",
            "E dey talk to himself for class, nobody dey grab wetin e dey talk",
            "E dey confuse person well well, explanations no get head",
            "The guy dey rush through slides like say thief dey pursue am",
            "E dey use big big grammar wey even dictionary no get",
            "I dey sleep for this guy's class because nothing dey enter head",
            "E dey just dey read slides like primary school teacher",
            "No be only confusing, e dey even scatter person brain join",
            "E dey teach like say e dey speak in tongues, who understand?",
            "The way e dey explain dey make person vex, nothing clear at all"
        ],
        "neutral": [
            "E dey follow textbook, nothing special but e dey go",
            "Explanations dey okay, no be die but no be kill",
            "Standard teaching, nothing to write home about",
            "E dey try sha, but sometimes e dey slow",
            "Lectures dey manageable, you go fit understand if you read well"
        ]
    },
    "Course Organisation": {
        "positive": [
            "Course dey well arranged, e no dey scatter at all",
            "E dey follow the syllabus like serious person",
            "No rush at all, e dey take time cover everything",
            "The course structure dey make sense, everything dey follow proper order",
            "E dey give us proper timetable, no surprises",
            "Course organization dey top notch, professional work",
            "Everything dey planned well, no disorganization",
            "E dey finish syllabus on time, no rushing last minute",
            "The way e dey arrange topics dey very logical",
            "Course dey run like smooth engine, no wahala"
        ],
        "negative": [
            "This course dey scatter scatter, no proper arrangement",
            "E dey jump from chapter 1 to chapter 10 back to chapter 3",
            "No proper syllabus, e dey just dey teach wetin e like",
            "Course dey disorganized well well, who understand this man?",
            "E dey rush everything for final weeks, like say exam dey tomorrow",
            "No proper outline, e dey just dey waka anyhow",
            "The course dey confused, topics dey jump up and down",
            "E dey disorganize everything, no proper flow",
            "Course organization dey zero, total nonsense",
            "E dey teach like say e no get plan, just dey do anyhow"
        ],
        "neutral": [
            "E dey follow the normal departmental syllabus",
            "Course dey go according to standard university format",
            "Topics dey covered as per the curriculum",
            "Nothing special, just normal course arrangement"
        ]
    },
    "Assessment and Grading": {
        "positive": [
            "This guy dey mark well well, no cheating",
            "Grading dey fair, if you work hard you go get your due",
            "E dey return scripts on time, no wahala",
            "Test questions dey based on wetin e teach for class",
            "E dey give helpful feedback, you go know where you miss am",
            "Marking dey transparent, no under the table business",
            "If you sabi the work, you go definitely pass",
            "E dey give proper rubrics before assignment",
            "Grading dey just, no favoritism at all",
            "Results dey come out quick, no long story"
        ],
        "negative": [
            "This man dey mark person wickedly, e dey fail people for nothing",
            "Grading dey unfair, e dey just dey anyhow",
            "Test questions dey different from wetin e teach for class",
            "E dey keep scripts till next semester, no return",
            "No proper grading, e dey just dey give marks randomly",
            "Person write well still fail, who understand this man?",
            "Grading dey harsh, e dey fail people like say na competition",
            "No rubric, e dey just dey mark as e like",
            "E dey fail people unnecessarily, heartless marking",
            "Grading dey biased, if e no like you, you don fail"
        ],
        "neutral": [
            "Grading dey follow normal university standard",
            "Two tests and one exam like every other course",
            "Marking dey okay, nothing special",
            "Standard assessment procedures"
        ]
    },
    "Lecturer Punctuality": {
        "positive": [
            "This guy dey come class early, dey wait for us",
            "E dey start class exactly on time, no delay",
            "Never late for once, very punctual man",
            "E dey respect time die, no African time",
            "Always punctual, professional behavior",
            "E dey come before 8am for 8am class",
            "Time keeper, e no dey waste student's time",
            "Very reliable with time, always available",
            "E dey start class sharp sharp, no wasting time",
            "Punctuality dey 100%, serious lecturer"
        ],
        "negative": [
            "This man dey come class late everyday",
            "E dey come 30 minutes after time don pass",
            "E dey cancel class like say e dey pure water",
            "Always late, time no dey important to am",
            "E dey waste our time with lateness",
            "Missed classes well well, no proper replacement",
            "E dey resume when e feel like, no respect for time",
            "Punctuality dey zero, e dey always come late",
            "E dey keep us waiting for nothing",
            "Time waster, always behind schedule"
        ],
        "neutral": [
            "E dey come roughly on time most times",
            "Occasional lateness but not too bad",
            "Timing dey okay, manageable",
            "Deports within reasonable time limits"
        ]
    },
    "Availability": {
        "positive": [
            "This guy dey always available for office hours",
            "E dey reply emails sharp sharp",
            "If you need help, e dey make time for you",
            "Very approachable, e dey welcome questions",
            "Office door dey always open for students",
            "E dey help struggling students well well",
            "You fit reach am anytime, e dey respond",
            "Very accessible lecturer, no wahala",
            "E dey attend to students like family",
            "Availability dey top notch, always there when needed"
        ],
        "negative": [
            "This man dey impossible to reach outside class",
            "Emails dey enter trash, e no dey reply",
            "Office hours dey for form only, door dey locked",
            "E dey shun students when we ask for help",
            "Impossible to see for office hours",
            "E dey cancel office hours without telling us",
            "When you need am, e no dey available",
            "Very unapproachable, e dey look students like trouble",
            "Help dey hard to get from this lecturer",
            "Availability dey zero, e dey run from students"
        ],
        "neutral": [
            "Available during normal office hours",
            "E dey see students if you book appointment",
            "Standard availability like other lecturers",
            "You fit reach am during consultation hours"
        ]
    },
    "Communication": {
        "positive": [
            "This guy dey communicate well well",
            "E dey reply emails fast, no dulling",
            "Announcements dey clear and timely",
            "E dey keep us informed about everything",
            "Communication dey professional and courteous",
            "E dey send updates regularly, no dull moment",
            "Very responsive to student inquiries",
            "E dey communicate like pro, no confusion",
            "Updates dey reach us on time",
            "Communication dey excellent, e dey carry us along"
        ],
        "negative": [
            "This man no dey reply emails at all",
            "Communication dey poor, e no dey inform us",
            "Announcements dey last minute, confusing",
            "E dey ignore messages like say we no exist",
            "Poor communication, students dey in the dark",
            "E dey keep important information to himself",
            "No proper channel of communication",
            "E dey fail to respond to urgent matters",
            "Communication dey zero, e no dey talk to us",
            "Unreliable information, confusing announcements"
        ],
        "neutral": [
            "E dey communicate through class rep",
            "Standard university communication channels",
            "Updates dey come through normal channels",
            "Communication dey okay, nothing special"
        ]
    },
    "Student Interaction": {
        "positive": [
            "This guy dey make class lively and interactive",
            "E dey encourage questions, nobody dey dull",
            "E dey respect students well well",
            "Class dey engaging, no dull moment",
            "E dey carry everybody along, no discrimination",
            "Very interactive lecturer, class dey sweet",
            "E dey listen to student contributions",
            "Classroom interaction dey top level",
            "E dey make students feel comfortable to talk",
            "Student interaction dey excellent, e dey relate well"
        ],
        "negative": [
            "This man dey intimidate students for class",
            "Class dey boring, e dey just lecture like radio",
            "E dey discourage questions, e dey vex when we ask",
            "No interaction at all, just one way traffic",
            "E dey ridicule students wey ask questions",
            "Class dey dull like graveyard, e no dey interact",
            "E dey ignore students wey raise hand",
            "Hostile environment, students dey fear to talk",
            "Unengaging lecturer, class dey sleep",
            "Student interaction dey zero, e no dey carry us along"
        ],
        "neutral": [
            "Interaction dey normal like other lecturers",
            "E dey entertain questions sometimes",
            "Standard lecture format with some interaction",
            "Nothing special, just normal class interaction"
        ]
    },
    "Use of Teaching Materials": {
        "positive": [
            "This guy dey give us better materials",
            "Slides dey comprehensive and well designed",
            "E dey share notes early, no last minute rush",
            "Materials dey up to date and helpful",
            "E dey use projector well, no wahala",
            "Slides dey clear and easy to understand",
            "E dey provide handouts and extra materials",
            "Teaching materials dey top notch",
            "E dey upload notes on time for us to read",
            "Materials dey very useful for revision"
        ],
        "negative": [
            "This man no dey give us any materials",
            "Slides dey old and outdated, like 1990s",
            "E dey refuse to share notes with us",
            "Materials dey poor, plenty errors",
            "No handouts, no nothing, just talk",
            "Slides dey blurry, you no fit read",
            "E dey force us to buy textbook",
            "Teaching materials dey zero, e no dey provide anything",
            "If you no buy handout, you no go get anything",
            "Materials dey inadequate for the course"
        ],
        "neutral": [
            "Standard slides like other courses",
            "E dey follow normal faculty materials",
            "Basic slides with bullet points",
            "Normal university materials"
        ]
    }
}

# Sarcasm templates - explicitly sarcastic comments
sarcasm_templates = {
    "Teaching Clarity": {
        "positive": [
            "Oh wow, this lecturer is a genius at making simple things sound like rocket science",
            "Brilliant! He managed to confuse us about things we already understood",
            "Amazing how he can make a 10-minute explanation last 45 minutes without saying anything",
            "What a talent for teaching - if by teaching you mean confusing students"
        ],
        "negative": [
            "Oh sure, because reading slides word-for-word is definitely 'teaching'",
            "Great job explaining things that weren't in the exam, really helpful",
            "Thanks for the 'clear' explanations that require a PhD to understand",
            "Wonderful how he manages to never actually explain anything"
        ]
    },
    "Course Organisation": {
        "positive": [
            "Perfect organization - if by perfect you mean chaotic and unpredictable",
            "Amazing how the syllabus is more of a suggestion than a plan",
            "Love how topics jump around like the lecturer has ADHD"
        ],
        "negative": [
            "Thanks for the surprise exam topics we never covered, really thoughtful",
            "Great planning - rushing everything in the last two weeks was genius",
            "Appreciate how the course outline was clearly written in invisible ink"
        ]
    },
    "Assessment and Grading": {
        "positive": [
            "Fair grading? Only if fair means random and arbitrary",
            "Love how grading rubrics are top secret until after you fail",
            "Thanks for testing us on things you never mentioned in class"
        ],
        "negative": [
            "Wonderful how the guy who wrote the best essay still failed",
            "Great feedback - 'needs improvement' really tells me everything",
            "Appreciate waiting 6 weeks for grades that were clearly randomly assigned"
        ]
    },
    "Lecturer Punctuality": {
        "positive": [
            "Always on time - if you consider 20 minutes late to be on time",
            "Professional - if by professional you mean making students wait for you",
            "Great time management - of the students' time, not his"
        ],
        "negative": [
            "Thanks for wasting 30 minutes of every class being late",
            "Love the surprise cancelled classes we find out about 5 minutes before",
            "Appreciate how he treats our time like it's worthless"
        ]
    },
    "Availability": {
        "positive": [
            "So available - if you can find him during the 5 minutes he's actually in office",
            "Great office hours - if the door wasn't always locked",
            "Very approachable - if by approachable you mean he runs away when he sees students"
        ],
        "negative": [
            "Thanks for never replying to emails, really shows dedication",
            "Love how office hours are just theoretical concepts",
            "Appreciate the help - or lack thereof - when we actually need it"
        ]
    },
    "Communication": {
        "positive": [
            "Clear communication - if you count silence as communication",
            "Great at informing us - of things that don't matter",
            "Professional emails - that never get replies"
        ],
        "negative": [
            "Thanks for the last-minute announcements, really helpful",
            "Love finding out about venue changes 10 minutes before class",
            "Appreciate the complete radio silence on important matters"
        ]
    },
    "Student Interaction": {
        "positive": [
            "Great interaction - if you count him talking to himself as interaction",
            "Very encouraging - of students to stay silent and avoid asking questions",
            "Welcoming environment - if you like being intimidated"
        ],
        "negative": [
            "Thanks for making students feel stupid for asking questions",
            "Love how he turns legitimate questions into insults",
            "Appreciate the hostile atmosphere that discourages participation"
        ]
    },
    "Use of Teaching Materials": {
        "positive": [
            "Great materials - if you count 10-year-old blurry slides as materials",
            "Comprehensive - if comprehensive means barely covering the basics",
            "Up-to-date - if by up-to-date you mean outdated by a decade"
        ],
        "negative": [
            "Thanks for making us buy textbooks you never reference",
            "Love the slides with more typos than actual content",
            "Appreciate the complete lack of any useful study materials"
        ]
    }
}

courses = [
    ("CSC410", "Special Computing"),
    ("CSC412", "Data Science & Big Data"),
    ("CSC406", "Cloud Computing Architectures"),
    ("CSC101", "Introduction to Computer Science"),
    ("CSC408", "Machine Learning & Neural Nets"),
    ("CSC302", "Database Design & Management"),
    ("CSC304", "Operating Systems Principles"),
    ("CSC201", "Data Structures and Algorithms"),
    ("CSC301", "Software Engineering"),
    ("CSC303", "Computer Networks")
]

lecturers = [
    "Dr. Okafor", "Dr. Adeyemi", "Prof. Martins",
    "Dr. Faith", "Dr. Pomele", "Prof. Balogun",
    "Dr. Chukwu", "Dr. (Mrs) Adeleke", "Dr. Ibrahim",
    "Prof. Okonkwo", "Dr. Nwosu", "Dr. Yusuf"
]

def generate_nigerian_slang_dataset(num_samples=2000):
    """
    Generates dataset with Nigerian slangs, informal expressions, and sarcasm.
    """
    aspect_list = list(nigerian_slang_templates.keys())
    records = []

    for i in range(num_samples):
        ccode, cname = random.choice(courses)
        lecturer = random.choice(lecturers)

        # Decide if this will be slang or sarcasm (70% slang, 30% sarcasm)
        is_sarcasm = random.random() < 0.3

        if is_sarcasm:
            # Generate sarcastic comment
            templates = sarcasm_templates
            # Sarcasm is typically negative, even when phrased as "positive"
            target_sentiment = random.choice(["negative", "negative", "negative", "positive"])
        else:
            # Generate slang comment
            templates = nigerian_slang_templates
            target_sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.35, 0.45, 0.20])[0]

        k = random.choices([1, 2, 3], weights=[0.30, 0.50, 0.20])[0]
        selected_aspects = random.sample(aspect_list, k)

        clauses = []
        clause_labels = []

        for idx, asp in enumerate(selected_aspects):
            if is_sarcasm:
                # For sarcasm, always use negative template regardless of target
                sentiment = "negative"
                phrase = random.choice(templates[asp].get("negative", templates[asp].get("positive", [""])))
            else:
                if target_sentiment == "neutral":
                    sentiment = "neutral"
                else:
                    sentiment = target_sentiment
                phrase = random.choice(templates[asp][sentiment])

            clauses.append(phrase)
            clause_labels.append((asp, sentiment))

        # Combine clauses
        if len(clauses) == 1:
            comment_text = clauses[0]
        elif len(clauses) == 2:
            connector = random.choice([", but ", ", however, ", ". Also, ", ", and "])
            comment_text = clauses[0] + connector + clauses[1]
        else:
            comment_text = clauses[0] + ", and " + clauses[1] + ", but " + clauses[2]

        # Determine rating based on sentiment
        if is_sarcasm:
            # Sarcasm is typically negative
            rating = random.choice([1, 2, 1, 2])
            doc_sentiment = "negative"
        else:
            pos_c = sum(1 for _, s in clause_labels if s == "positive")
            neg_c = sum(1 for _, s in clause_labels if s == "negative")
            if pos_c > neg_c:
                doc_sentiment = "positive"
                rating = random.choice([4, 5, 4])
            elif neg_c > pos_c:
                doc_sentiment = "negative"
                rating = random.choice([1, 2, 1])
            else:
                doc_sentiment = "neutral"
                rating = 3

        records.append({
            "id": f"slang-{i + 1}",
            "lecturer_name": lecturer,
            "course": cname,
            "course_code": ccode,
            "rating": rating,
            "comment": comment_text,
            "document_sentiment": doc_sentiment,
            "aspect_labels": json.dumps(clause_labels),
            "source": "nigerian_slang"
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    print("Generating Nigerian slang and sarcasm dataset...")
    df = generate_nigerian_slang_dataset(num_samples=2000)
    df.to_csv("data/nigerian_slang_dataset.csv", index=False)
    print(f"Generated {len(df)} Nigerian slang/sarcasm records")
    print(f"Saved to data/nigerian_slang_dataset.csv")
    print("\nDocument sentiment distribution:")
    print(df["document_sentiment"].value_counts())
    print("\nAspect distribution:")
    aspect_counts = {}
    for _, row in df.iterrows():
        labels = json.loads(row["aspect_labels"])
        for asp, sent in labels:
            if asp not in aspect_counts:
                aspect_counts[asp] = {"positive": 0, "negative": 0, "neutral": 0}
            aspect_counts[asp][sent] += 1
    for asp, counts in aspect_counts.items():
        print(f"  {asp}: {counts}")
