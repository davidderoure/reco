# app.py - Flask web server for Story Recommender Demo (Updated)

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

# Import your recommender system
from recommender import StoryRecommender, AnalyticsEvent

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Initialize recommender with configurable slots
recommender = StoryRecommender(
    event_half_life_days=30.0,
    connectedness_half_life_days=14.0,
    transition_window_minutes=1440.0,
    recommendation_config={
        'content': 2,
        'collaborative': 2,
        'topical': 1,
        'wildcard': 1
    }
)

# Add sample stories with tags (no themes)
def initialize_stories():
    # Ancient artifacts (6 stories)
    recommender.add_story("story1", "The Alfred Jewel", 
                         ["ancient", "mysterious", "royal", "craftsmanship"])
    recommender.add_story("story4", "The Scorpion Macehead", 
                         ["ancient", "Egyptian", "powerful", "discovery"])
    recommender.add_story("story8", "The Parian Marble", 
                         ["ancient", "chronological", "scholarly", "timeless"])
    recommender.add_story("story11", "The Minoan Snake Goddess", 
                         ["ancient", "mystical", "feminine", "ritual"])
    recommender.add_story("story12", "The Roman Mosaic", 
                         ["ancient", "artistic", "domestic", "preserved"])
    recommender.add_story("story13", "The Ure Greek Vase", 
                         ["ancient", "athletic", "celebration", "beauty"])
    
    # Natural history (4 stories)
    recommender.add_story("story2", "The Last Dodo", 
                         ["natural", "extinct", "haunting", "loss"])
    recommender.add_story("story7", "Tradescant's Ark", 
                         ["natural", "curious", "wondrous", "collection"])
    recommender.add_story("story14", "The Ichthyosaur", 
                         ["natural", "prehistoric", "marine", "fossilized"])
    recommender.add_story("story15", "The Giant Irish Deer", 
                         ["natural", "magnificent", "ice-age", "extinct"])
    
    # Medieval (4 stories)
    recommender.add_story("story3", "Guy Fawkes' Lantern", 
                         ["medieval", "conspiracy", "history", "rebellion"])
    recommender.add_story("story6", "The Abingdon Sword", 
                         ["medieval", "warrior", "crafted", "legendary"])
    recommender.add_story("story16", "The Illuminated Manuscript", 
                         ["medieval", "sacred", "illustrated", "devotional"])
    recommender.add_story("story17", "The Lewis Chessmen", 
                         ["medieval", "carved", "strategic", "mysterious"])
    
    # Cultural (4 stories)
    recommender.add_story("story5", "Powhatan's Mantle", 
                         ["cultural", "ceremonial", "heritage", "connection"])
    recommender.add_story("story9", "Ceremonial Axes", 
                         ["cultural", "ritual", "spiritual", "ancestral"])
    recommender.add_story("story18", "The Shrunken Heads", 
                         ["cultural", "transformative", "warrior", "ritual"])
    recommender.add_story("story19", "The Samurai Armor", 
                         ["cultural", "honor", "protective", "disciplined"])
    
    # Scientific (3 stories)
    recommender.add_story("story10", "Einstein's Blackboard", 
                         ["scientific", "genius", "lecture", "revelation"])
    recommender.add_story("story20", "The Astrolabe", 
                         ["scientific", "navigational", "astronomical", "precise"])
    recommender.add_story("story21", "Carroll's Camera", 
                         ["scientific", "photographic", "innovative", "capturing"])
    
    # Artistic (3 stories)
    recommender.add_story("story22", "The Light of the World", 
                         ["artistic", "symbolic", "glowing", "spiritual"])
    recommender.add_story("story23", "Michelangelo's Drawing", 
                         ["artistic", "masterful", "anatomical", "renaissance"])
    recommender.add_story("story24", "Islamic Ceramic Bowl", 
                         ["artistic", "geometric", "calligraphic", "luminous"])
    
    # Literary (1 story)
    recommender.add_story("story25", "Shakespeare's First Folio", 
                         ["literary", "dramatic", "immortal", "eloquent"])

initialize_stories()

# Story content
STORY_CONTENT = {
    "story1": """
        In the Ashmolean Museum, behind glass that has protected it for centuries, 
        lies the Alfred Jewel—a masterpiece of Anglo-Saxon craftsmanship. Barely 
        larger than a thumb, this golden artifact bears the inscription "AELFRED MEC 
        HEHT GEWYRCAN"—Alfred ordered me to be made.
        
        King Alfred the Great commissioned this jewel over a thousand years ago, perhaps 
        as a pointer for reading sacred texts. The enamel figure gazes out with knowing 
        eyes, holding flowering rods, forever frozen in a moment of medieval artistry. 
        Gold, enamel, and rock crystal—materials that would outlast kingdoms.
        
        Discovered in 1693 in a Somerset field, the jewel had waited centuries in the 
        earth. What stories could it tell? Of the king who held it, of the craftsman 
        who shaped it, of the battles and books and prayers it witnessed. In your 
        reflection in its glass case, you become part of its endless story—another 
        pair of eyes that has beheld its beauty, another moment in its long existence.
    """,
    "story2": """
        The last complete dodo in existence stands in the Oxford University Museum of 
        Natural History—or rather, what remains of one does. A head. A foot. Fragments 
        of a bird that vanished from the world in the 1660s.
        
        Once, dodos waddled fearlessly through the forests of Mauritius, an island where 
        they had no predators. They couldn't fly—why would they need to? But then humans 
        arrived, bringing rats and pigs and hunger. Within decades, the dodo was gone.
        
        This specimen came from the collection of John Tradescant, who displayed it as 
        a curiosity. When it began to rot, most of it was destroyed. Only these pieces 
        were saved. Now they stand as a memorial to extinction, to the fragility of 
        species, to the worlds we lose without even knowing what we had.
    """,
    "story3": """
        A simple lantern hangs in the Ashmolean Museum, its metal tarnished with age. 
        On the night of November 4, 1605, Guy Fawkes carried this lantern into the 
        cellars beneath Parliament, where 36 barrels of gunpowder waited in the darkness.
        
        He never got to light the fuse. Guards discovered him, and the Gunpowder Plot 
        failed. But this lantern witnessed the moment when English history could have 
        taken a catastrophically different turn. What if Fawkes had succeeded? What if 
        Parliament had exploded, killing King James and his government?
        
        The lantern holds no answers, only the weight of what didn't happen. We celebrate 
        Guy Fawkes Night every November, burning effigies, lighting fireworks—remembering 
        the plot that failed, the explosion that never came, the light that never ignited 
        in this unassuming lamp.
    """,
    "story4": """
        The Scorpion Macehead is one of the oldest royal artifacts in existence, carved 
        over 5,000 years ago to commemorate a king whose name we know only by his symbol: 
        a scorpion. He ruled Egypt before the pharaohs, before the pyramids, at the very 
        dawn of civilization.
        
        The macehead shows the Scorpion King digging a canal, perhaps performing a 
        ceremony to bless the irrigation systems that would feed his people. Around him, 
        symbols and figures tell a story we can barely decipher—a language of power and 
        ritual from an age before writing as we know it.
        
        Discovered at Hierakonpolis, the ancient city of the falcon, this artifact 
        connects us to a world so distant that even the ancient Egyptians would have 
        considered it ancient. The scorpion, symbol of death and protection, watches 
        over a king whose deeds are remembered only in stone.
    """,
    "story5": """
        Powhatan's Mantle hangs in the Ashmolean, a deerskin cloak decorated with 
        shells in patterns that speak of power and connection. It may have belonged 
        to Powhatan himself, father of Pocahontas, paramount chief of the Tsenacommacah 
        alliance in 17th-century Virginia.
        
        The shells form figures—human shapes, circles, patterns whose meanings have 
        been debated for centuries. Were they maps? Symbols of authority? Records of 
        alliances? The mantle traveled from Virginia to England in the early 1600s, 
        a gift or a trophy from a time when two worlds collided.
        
        To stand before it is to feel the weight of that collision—the Powhatan 
        Confederacy that thrived before English colonization, the cultures that would 
        be devastated, the stories that would be lost. The mantle survived. In its 
        shells and symbols, it carries memories of a world that was.
    """,
    "story6": """
        The Abingdon Sword, discovered in the Thames, is a weapon from the 15th century 
        that has been perfectly preserved by the river mud. Its blade still sharp, its 
        grip still intact, it looks as though its owner might return to claim it at any 
        moment.
        
        Who dropped this sword into the Thames? A knight fleeing from battle? A 
        ceremony of some kind? An accident? We will never know. But the sword carries 
        the marks of use—this was no ceremonial piece. It saw combat, defending or 
        attacking, wielded by someone whose name has been lost to time.
        
        Medieval swords were more than weapons; they were symbols of status, objects 
        of reverence, sometimes given names and attributed almost mystical properties. 
        This sword has no name now, only the river's memory of the moment it slipped 
        from someone's grasp and sank into the dark water, waiting five hundred years 
        to be found.
    """,
    "story7": """
        Tradescant's Ark was one of the world's first public museums, a cabinet of 
        curiosities assembled by John Tradescant and his son in 17th-century London. 
        The collection mixed the exotic and the impossible: a dodo, a dragon's egg, 
        Guy Fawkes' lantern, Powhatan's mantle, things from the edges of the known world.
        
        The Tradescants were gardeners to royalty who became obsessed with collecting. 
        They didn't distinguish between natural specimens and mythical objects, between 
        science and wonder. Their catalogue listed things like "Edward the Confessor's 
        knit gloves" alongside genuine treasures from around the globe.
        
        When the collection came to Oxford, it became the foundation of the Ashmolean 
        Museum. The Ark is scattered now, its pieces in different collections, but its 
        spirit lives on—that sense of wonder at the world's strangeness, that desire 
        to gather and understand and marvel at everything we can find.
    """,
    "story8": """
        The Parian Marble is a chronological table carved on the Greek island of Paros 
        around 264 BCE. It lists the most important events in Greek history, from the 
        legendary King Cecrops in 1581 BCE to the archonship of Diognetus in 264 BCE—
        over 1,300 years of history etched in stone.
        
        But what the anonymous scribes considered "important" reveals as much about 
        them as about history. Poets winning competitions receive as much attention as 
        military victories. The invention of the flute is carefully dated. Cultural 
        achievements matter more than conquests.
        
        The marble came to Oxford in the 17th century, already ancient, already 
        fragmentary. What remains is both a historical document and a reminder of 
        how we choose what to remember. History is not just what happened, but what 
        we decide was worth recording, worth preserving, worth carrying forward through 
        the centuries.
    """,
    "story9": """
        The ceremonial axes in the Pitt Rivers Museum come from cultures across the 
        world, each one a masterpiece of craftsmanship never meant to cut wood. Jade 
        axes from China, stone axes from the Pacific, metal axes from Africa—objects 
        of power and prestige.
        
        These axes tell us that humans have always understood the difference between 
        the practical and the symbolic. An axe can be a tool, or it can be a statement. 
        It can be traded for wealth, given as a gift to forge alliances, buried with 
        the dead as a companion in the afterlife.
        
        In many cultures, the axe represented authority—the power to cut, to divide, 
        to decide. But a ceremonial axe, too beautiful or precious to use, becomes 
        something else entirely: a promise of power, a symbol of what might be done 
        but isn't, potential energy frozen in jade or stone or bronze.
    """,
    "story10": """
        Einstein's blackboard hangs in the Museum of the History of Science, still 
        covered with the chalk marks from a lecture he gave at Oxford in 1931. The 
        equations explore whether the universe is expanding or static, calculations 
        about the nature of reality itself.
        
        Einstein never erased the board, and no one else dared to. These are his 
        handwriting, his thoughts mid-formation, his attempt to explain the cosmos to 
        an Oxford audience. Some of the equations would later prove to be wrong—Einstein 
        himself would abandon some of these ideas. But they capture a moment in the 
        development of cosmology, a great mind working through the greatest questions.
        
        To look at the blackboard is to see thinking made visible. Not the polished 
        final theory, but the process—the crossing out, the revision, the moment of 
        explanation. Science as it actually happens, messy and beautiful, written in 
        chalk that could have been wiped away but wasn't.
    """,
    "story11": """
        The Minoan Snake Goddess stands barely seven inches tall, but she commanded 
        the religious life of Bronze Age Crete. Her arms raised, a snake in each hand, 
        her expression serene—she is authority and mystery combined.
        
        Snakes represented renewal in Minoan culture, shedding their skin to be born 
        anew. This goddess, whoever she was—deity, priestess, or symbolic figure—held 
        that power of transformation. The Minoans worshipped in caves and on mountaintops, 
        their rituals lost to time, their writing still undeciphered.
        
        Found at Knossos, the great palace complex, this figurine survived the 
        catastrophic collapse of Minoan civilization around 1450 BCE. Earthquakes, 
        tsunamis, or invasion—something ended their world. But the snake goddess 
        remained, buried under the ruins, waiting thousands of years to be found and 
        reveal a glimpse of a lost religion.
    """,
    "story12": """
        A Roman mosaic floor, lifted carefully from its original location and preserved 
        in the museum, shows a scene of daily life from nearly 2,000 years ago. Figures 
        in togas, geometric patterns, the kind of floor a wealthy Roman family would 
        have walked across every day.
        
        What's remarkable is how ordinary it is. This wasn't a grand public monument 
        but someone's home. Children played on these stones. Servants swept them. Dinner 
        parties happened here, business was conducted, lives were lived.
        
        The mosaic survived because it was buried and forgotten, protected by the earth 
        from centuries of weather and warfare. Now, lifted into the museum, it becomes 
        art. But it was never meant to be art—it was meant to be a floor, beautiful but 
        functional, a reminder that the Romans filled their daily lives with beauty in 
        ways we've mostly forgotten how to do.
    """,
    "story13": """
        The Ure Greek Vase depicts athletes in motion—runners, wrestlers, discus throwers 
        frozen in red and black. Created around 500 BCE, it celebrates the games that 
        brought Greek city-states together in competition rather than war.
        
        The artists who painted these vases were craftsmen, not usually signing their 
        work, but their skill is undeniable. Every muscle is understood, every movement 
        captured with economy and grace. These were mass-produced items, everyday objects 
        for storing oil or wine, but they were made beautiful almost casually, as if 
        beauty was just something that happened.
        
        The vase reminds us that the ancient Greeks didn't separate art from life the 
        way we do. Athletes were beautiful, so depicting them was natural. Competitions 
        were sacred, so commemorating them was proper. The vase held oil once, was used 
        and handled and valued, and now it holds something else: our wonder at a culture 
        that could make even utilitarian objects sing.
    """,
    "story14": """
        The ichthyosaur skeleton stretches across the museum wall, a marine reptile 
        from the Jurassic seas. It looks like a dolphin but isn't one—this is 
        convergent evolution, two very different animals developing similar shapes to 
        solve the same problem: how to move through water efficiently.
        
        Ichthyosaurs ruled the oceans when dinosaurs ruled the land, giving birth to 
        live young in the water, hunting fish and squid. This particular specimen was 
        fossilized in such detail that you can see the outline of its body, even traces 
        of skin and the eye ring.
        
        It died 180 million years ago, settling to the sea floor where sediment slowly 
        covered it. The seas dried up, the land rose, and eventually someone with a 
        hammer and an eye for fossils split the rock and found this perfect time capsule. 
        Now it swims forever on a museum wall, suspended in stone and story.
    """,
    "story15": """
        The Giant Irish Deer towers over museum visitors, its antlers spanning twelve 
        feet—the largest of any deer that ever lived. It roamed Ice Age Europe and 
        Ireland until about 7,700 years ago, when the last of them died out.
        
        Those massive antlers weren't just for show. They were weapons, displays of 
        fitness, signals to rivals and potential mates. But they were also a burden. 
        Growing them required enormous amounts of calcium and energy. When climate 
        changed and food became scarce, those magnificent antlers may have contributed 
        to the species' extinction.
        
        The skeleton is humbling. Something so powerful, so perfectly adapted to its 
        world, could still vanish. The Irish Deer was successful for hundreds of 
        thousands of years, and then it wasn't. A reminder that dominance is temporary, 
        that even the giants fall, that evolution creates wonders and then erases them.
    """,
    "story16": """
        The illuminated manuscript glows with gold leaf and brilliant pigments, every 
        page a work of art. Medieval monks spent years creating these books, copying 
        religious texts by hand and decorating them with intricate designs and miniature 
        paintings.
        
        The work was sacred—literally. These weren't just books but prayers made visible, 
        offering to God in the form of beauty and time. A single page might take weeks 
        to complete. The blue came from crushed lapis lazuli, more expensive than gold. 
        The red from insects. Every color was precious, every line deliberate.
        
        To look at an illuminated manuscript is to see faith transformed into art. The 
        monks who made these books believed they were creating something eternal, and 
        they were right. Centuries later, their devotion still shines from the page, 
        gold catching the light, colors as vivid as the day they were painted.
    """,
    "story17": """
        The Lewis Chessmen are among the most famous chess pieces in the world—carved 
        from walrus ivory in the 12th century and discovered buried on the Isle of Lewis 
        in Scotland in 1831. Why were they buried? Who hid them? We still don't know.
        
        The pieces are wonderfully expressive. The queens hold their faces in their hands, 
        looking concerned. The bishops look stern. The knights on horseback are ready 
        for battle. And the rooks are berserkers, biting their shields in battle fury.
        
        They were probably made in Norway and represent the spread of chess across 
        medieval Europe—a game from Persia that became a metaphor for medieval warfare 
        and strategy. Someone carried these pieces across the sea and buried them for 
        safekeeping. They waited eight centuries in the sand to tell us about medieval 
        life, about trade routes, about a game that outlasted kingdoms.
    """,
    "story18": """
        The shrunken heads in the Pitt Rivers Museum are among its most controversial 
        objects. They come from the Shuar people of Ecuador and Peru, created through 
        a sacred ritual that transformed enemies into protective spirits.
        
        The practice was spiritual, not savage. When a warrior killed an enemy, the 
        head was preserved through an elaborate process to trap the spirit and prevent 
        it from seeking revenge. The head became small, the features distorted but still 
        recognizable, a powerful object of ritual significance.
        
        European collectors turned these sacred objects into curiosities, stripping away 
        their meaning. The museum now struggles with how to display them—as ethnographic 
        specimens, as art, as human remains? They force us to confront uncomfortable 
        questions about cultural respect, about colonialism, about what belongs in museums 
        and what should be returned.
    """,
    "story19": """
        The samurai armor stands complete, every piece intricately laced together with 
        silk cords, lacquered and decorated. It was meant to protect, yes, but also to 
        intimidate and to honor. A samurai's armor was an extension of his soul.
        
        Each component had a name and purpose. The helmet's crest identified the warrior's 
        clan. The mask protected the face while creating a fearsome appearance. The 
        shoulder guards allowed movement while deflecting sword blows. The whole suit 
        could weigh 40 pounds, yet samurai trained until they could move in it like water.
        
        This armor never saw battle but was worn for ceremonies, a reminder of the 
        warrior's code even in peacetime. Bushido—the way of the warrior—valued honor 
        above life, duty above comfort. The armor embodies that philosophy: beautiful, 
        functional, representing an ideal that transcended its practical purpose.
    """,
    "story20": """
        The astrolabe is a mechanical map of the heavens, a device that could tell time, 
        navigate by stars, and predict celestial events. Islamic scholars perfected its 
        design in the medieval period, creating instruments of extraordinary precision 
        and beauty.
        
        Holding an astrolabe is like holding the cosmos. The rotating plates represent 
        the movement of stars. The sights align with celestial bodies. With practice, 
        you could determine your latitude, find the direction of Mecca for prayer, or 
        calculate the time of sunrise.
        
        This particular astrolabe is decorated with Arabic inscriptions and geometric 
        patterns, science and art inseparable. It represents a time when Islamic 
        civilization led the world in astronomy, mathematics, and navigation—when the 
        stars were not distant lights but a practical guide to understanding our place 
        in the universe.
    """,
    "story21": """
        Lewis Carroll's camera sits in a display case, a wooden box with brass fittings, 
        the tool the mathematician and author used to pursue his other passion: 
        photography. Carroll was one of the great Victorian photographers, capturing 
        portraits of children with a sensitivity that was revolutionary.
        
        Photography was new in Carroll's time, magical and chemical and strange. The 
        process required subjects to stay perfectly still for long exposures. Carroll 
        would pose children naturally, telling them stories, making them comfortable, 
        creating images that felt alive despite the technical constraints.
        
        The camera connects Carroll's two worlds—the mathematical precision required 
        for photography and the imaginative stories like Alice in Wonderland. Both 
        required seeing the world differently, catching something real and making it 
        wondrous, whether through a lens or through words.
    """,
    "story22": """
        "The Light of the World" by Holman Hunt shows Christ holding a lantern, knocking 
        on a door overgrown with weeds—a door with no handle on the outside. The 
        symbolism is clear: the door is the human soul, which can only be opened from 
        within.
        
        Hunt was a Pre-Raphaelite, part of a movement that rejected the industrial age's 
        ugliness and sought truth in nature and symbolism. He painted at night to capture 
        the exact quality of lantern light, obsessive in his dedication to authentic detail.
        
        The painting became one of the most reproduced images in Victorian England, a 
        message of faith and free will. Christ waits patiently at the door but will not 
        force entry. The light he carries illuminates, reveals, offers hope—but the 
        choice to open the door belongs to each person alone.
    """,
    "story23": """
        Michelangelo's drawing shows a human form in perfect anatomical detail—every 
        muscle understood, every proportion calculated. Renaissance artists studied corpses 
        to understand the body, seeking truth beneath the skin, seeing science and art 
        as inseparable pursuits.
        
        Michelangelo believed that the artist's job was to free the figure already present 
        in the marble or on the page. Drawing was thinking, a way of understanding the 
        divine geometry of the human form. These sketches were never meant for public 
        display but were studies, explorations, the master teaching himself.
        
        To see a Michelangelo drawing is to see genius in process. Not the finished 
        masterpiece but the working out, the learning, the moment of understanding 
        captured in chalk or ink. It reminds us that even Michelangelo had to practice, 
        had to study, had to work to achieve what seemed like effortless perfection.
    """,
    "story24": """
        The Islamic ceramic bowl is a masterpiece of geometric and calligraphic design, 
        created in 13th-century Persia. Blue and white patterns interlock with mathematical 
        precision, Arabic script flowing around the rim in a verse from the Quran.
        
        Islamic art avoided depicting humans or animals in religious contexts, instead 
        developing geometric patterns of extraordinary complexity. These patterns weren't 
        just decoration but meditation on the infinite—mathematical representations of 
        divine order, endless repetitions suggesting the eternal.
        
        The bowl was functional, meant to hold food or water, but it was also a reminder 
        of the sacred in the everyday. Every meal could be a spiritual act, beauty present 
        in every moment of life. The potters who created these objects saw no division 
        between craft and art, between utility and transcendence.
    """,
    "story25": """
        Shakespeare's First Folio, published in 1623, seven years after his death, 
        preserved 36 of his plays. Without this book, we would have lost "Macbeth," 
        "The Tempest," "Julius Caesar," and more—half of Shakespeare's work existed 
        nowhere else.
        
        The actors who knew Shakespeare, John Heminges and Henry Condell, gathered the 
        scripts and published them "to keep the memory of so worthy a friend and fellow 
        alive." They couldn't have known they were preserving the most influential body 
        of work in English literature.
        
        The Folio is enormous, heavy, expensive—it was meant to proclaim Shakespeare's 
        importance. The portrait on the frontispiece shows a man with a receding hairline 
        and an enigmatic expression, the only image we have with any claim to accuracy. 
        The book opens with a poem by Ben Jonson: "He was not of an age, but for all time." 
        Nearly 400 years later, that prophecy holds true.
    """
}

def get_user_id():
    """Get or create user ID from session"""
    if 'user_id' not in session:
        session['user_id'] = f"demo_user_{datetime.now().timestamp()}"
    return session['user_id']

@app.route('/')
def index():
    """Home page - browse by tags and story recommendations"""
    user_id = get_user_id()
    
    # Get user profile if exists
    user = recommender.users.get(user_id)
    
    # Get user stats
    stats = {
        'stories_read': len(user.viewed_stories) if user else 0,
        'stories_completed': len([1 for pct, _ in user.story_progress.values() if pct >= 100]) if user else 0,
        'bookmarks': len(user.bookmarked_stories) if user else 0,
        'last_completed': None
    }
    
    if user and user.last_completed_story and user.last_completed_story in recommender.stories:
        stats['last_completed'] = recommender.stories[user.last_completed_story].title
    
    # Get all available tags
    all_tags = sorted(list(recommender.available_tags))
    
    return render_template('index.html', 
                         stats=stats,
                         user_id=user_id,
                         all_tags=all_tags)

@app.route('/recommendations')
def recommendations():
    """Show personalized recommendations"""
    user_id = get_user_id()
    
    # Get recommendations with method tracking
    recs = recommender.get_recommendations(user_id, n_recommendations=6)
    
    # Prepare recommendation data
    rec_data = []
    for story_id, score, method, slot_position in recs:
        story = recommender.stories[story_id]
        
        # Get reasons for recommendation based on method
        reasons = []
        
        if method == 'content':
            reasons.append("Based on stories you connected with")
        elif method == 'collaborative':
            reasons.append("Popular with readers like you")
        elif method == 'topical':
            reasons.append("Trending or newly added")
        elif method == 'wildcard':
            reasons.append("Something different to explore")
        elif method == 'sequence':
            reasons.append("Great follow-up to your last story")
        
        # Check if it's a good follow-up to last completed story
        user = recommender.users.get(user_id)
        if user and user.last_completed_story:
            last_story = recommender.stories.get(user.last_completed_story)
            if last_story and story_id in last_story.best_next_stories:
                effect = last_story.best_next_stories[story_id]
                reasons.append(f"Works well after '{last_story.title}'")
        
        # Check connectedness
        if story.avg_connectedness and story.avg_connectedness >= 4:
            reasons.append(f"High reader connection (avg: {story.avg_connectedness:.1f}/5)")
        
        rec_data.append({
            'id': story_id,
            'title': story.title,
            'tags': story.tags,
            'score': score,
            'method': method,
            'slot_position': slot_position,
            'reasons': reasons,
            'avg_connectedness': story.avg_connectedness
        })
    
    return render_template('recommendations.html', recommendations=rec_data)

@app.route('/browse_tag/<tag>')
def browse_tag(tag):
    """Browse stories by tag"""
    user_id = get_user_id()
    
    # Record tag search event
    recommender.user_searched_tag(user_id, tag, datetime.now())
    
    # Get stories with this tag
    stories_with_tag = [
        story for story in recommender.stories.values()
        if tag in story.tags
    ]
    
    return render_template('browse_tag.html', tag=tag, stories=stories_with_tag)

@app.route('/story/<story_id>')
def view_story(story_id):
    """View a story"""
    user_id = get_user_id()
    
    if story_id not in recommender.stories:
        return redirect(url_for('index'))
    
    story = recommender.stories[story_id]
    content = STORY_CONTENT.get(story_id, "Story content not available.")
    
    # Record view event
    recommender.user_viewed_story(user_id, story_id, datetime.now())
    
    # Check if already completed
    user = recommender.users.get(user_id)
    already_completed = False
    if user and story_id in user.story_progress:
        completion_pct, _ = user.story_progress[story_id]
        already_completed = completion_pct >= 100
    
    # Store story view time in session for progress tracking
    session[f'story_start_{story_id}'] = datetime.now().isoformat()
    
    return render_template('story.html', 
                         story=story, 
                         content=content,
                         already_completed=already_completed)

@app.route('/story_progress/<story_id>', methods=['POST'])
def story_progress(story_id):
    """Record story reading progress when user leaves the story"""
    user_id = get_user_id()
    
    # Get completion percentage from form (V1 naming: read_percent)
    read_percent = int(float(request.form.get('completion_percentage', 0)))
    
    # Record progress event using V1-aligned method
    recommender.user_read_story(user_id, story_id, read_percent, datetime.now())
    
    # If 100% complete, redirect to questions
    if read_percent >= 100:
        return redirect(url_for('story_questions', story_id=story_id))
    else:
        # Partial completion, go back to recommendations
        return redirect(url_for('recommendations'))

@app.route('/story_questions/<story_id>')
def story_questions(story_id):
    """Show post-reading questions"""
    story = recommender.stories.get(story_id)
    if not story:
        return redirect(url_for('index'))
    
    user_id = get_user_id()
    user = recommender.users.get(user_id)
    
    # Check if already bookmarked
    already_bookmarked = user and story_id in user.bookmarked_stories
    
    return render_template('story_questions.html', 
                         story=story,
                         already_bookmarked=already_bookmarked)

@app.route('/submit_question/<story_id>', methods=['POST'])
def submit_question(story_id):
    """Submit answer to a question"""
    user_id = get_user_id()
    question_number = int(request.form['question_number'])
    response = int(request.form['response'])
    
    # Record question response using V1-aligned method
    recommender.user_answered_question(
        user_id, 
        story_id, 
        response, 
        question_number, 
        datetime.now()
    )
    
    # Return to questions page to allow answering more questions
    return redirect(url_for('story_questions', story_id=story_id))

@app.route('/bookmark_story/<story_id>', methods=['POST'])
def bookmark_story(story_id):
    """Bookmark a story"""
    user_id = get_user_id()
    
    # Record bookmark event using V1-aligned method
    recommender.user_bookmarked_story(user_id, story_id, datetime.now())
    
    # Redirect back to where the user came from
    return redirect(request.referrer or url_for('recommendations'))

@app.route('/insights')
def insights():
    """Show insights about sequences and patterns"""
    user_id = get_user_id()
    
    # Get sequence insights
    insights_data = recommender.get_sequence_insights(user_id)
    
    # Get user-specific data
    user = recommender.users.get(user_id)
    user_data = None
    
    if user:
        # Get tag preferences
        tag_scores = user._get_decayed_tag_scores(datetime.now())
        
        # Get recommendation method breakdown
        method_counts = defaultdict(int)
        method_selected_counts = defaultdict(int)
        for rec in user.recommendations_shown:
            method_counts[rec.method] += 1
            if rec.selected:
                method_selected_counts[rec.method] += 1
        
        user_data = {
            'tag_scores': sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            'sequences': insights_data.get('user_sequences', [])[-10:],
            'question_responses': user.question_responses[-20:],
            'method_counts': dict(method_counts),
            'method_selected_counts': dict(method_selected_counts),
            'ignore_counts': insights_data.get('user_ignore_counts', {})
        }
    
    return render_template('insights.html', 
                         insights=insights_data,
                         user_data=user_data)

@app.route('/export_state')
def export_state():
    """Export full analytical state as JSON (for testing and analysis)"""
    state = recommender.save_state(mode="full")
    return jsonify(state)

@app.route('/export_daily/<date_str>')
def export_daily(date_str):
    """
    Export analytical state for a specific date.
    Example: /export_daily/2024-01-15
    """
    try:
        target_date = datetime.fromisoformat(date_str)
        start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        state = recommender.save_state(mode="full", start_date=start_date, end_date=end_date)
        return jsonify(state)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

@app.route('/checkpoint')
def checkpoint():
    """Get operational checkpoint (lightweight, for fault tolerance)"""
    state = recommender.save_state(mode="operational")
    return jsonify(state)

@app.route('/save_checkpoint', methods=['POST'])
def save_checkpoint():
    """Save operational checkpoint to file"""
    try:
        filepath = recommender.save_operational_checkpoint()
        return jsonify({'success': True, 'filepath': filepath})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_analytical_export', methods=['POST'])
def save_analytical_export():
    """Save analytical export to file (daily or on-demand)"""
    try:
        # Optional: get date range from request
        data = request.get_json() if request.is_json else {}
        start_date = None
        end_date = None
        
        if 'start_date' in data:
            start_date = datetime.fromisoformat(data['start_date'])
        if 'end_date' in data:
            end_date = datetime.fromisoformat(data['end_date'])
        
        filepath = recommender.export_analytical_state(
            start_date=start_date,
            end_date=end_date
        )
        return jsonify({'success': True, 'filepath': filepath})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reset')
def reset():
    """Reset the demo (clear session)"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
