import random

# Configuration
TRAIN_LINES = 50000  # Increased count again for new data types
VALID_LINES = 100
seen_pairs = set()

def get_math():
    ops = ['+', '-', '*']
    op = random.choice(ops)
    a = random.randint(0, 500) 
    b = random.randint(0, 500)
    
    if op == '+':
        return f"what is {a} + {b}?", str(a + b)
    elif op == '-':
        if a < b: a, b = b, a
        return f"what is {a} - {b}?", str(a - b)
    else:
        a_small = random.randint(0, 100)
        b_small = random.randint(0, 100)
        return f"calculate {a_small} * {b_small}", str(a_small * b_small)

def get_comparison():
    a = random.randint(0, 1000)
    b = random.randint(0, 1000)
    while a == b:
        b = random.randint(0, 1000)
    
    if random.random() > 0.5:
        return f"is {a} greater than {b}?", "yes" if a > b else "no"
    else:
        return f"is {a} less than {b}?", "yes" if a < b else "no"

def get_sequence():
    start = random.randint(1, 2000)
    step = random.randint(1, 100)
    seq = [start, start + step, start + 2*step]
    return f"complete the sequence {seq[0]}, {seq[1]}, {seq[2]}...", str(start + 3*step)

def get_antonym():
    pairs = [
        ("hot", "cold"), ("up", "down"), ("left", "right"), ("day", "night"),
        ("black", "white"), ("happy", "sad"), ("big", "small"), ("fast", "slow"),
        ("hard", "soft"), ("wet", "dry"), ("start", "stop"), ("win", "lose"), 
        ("life", "death"), ("rich", "poor"), ("young", "old"), ("tall", "short"),
        ("clean", "dirty"), ("empty", "full"), ("near", "far"), ("dark", "light"),
        ("heavy", "light"), ("open", "closed"), ("in", "out"), ("love", "hate"),
        ("good", "bad"), ("true", "false"), ("smooth", "rough"), ("thick", "thin"),
        ("laugh", "cry"), ("brave", "cowardly"), ("friend", "enemy"), ("first", "last"),
        ("buy", "sell"), ("give", "take"), ("push", "pull"), ("early", "late"),
        ("strong", "weak"), ("loud", "quiet"), ("smart", "stupid"), ("enter", "exit"),
        ("always", "never"), ("top", "bottom"), ("front", "back"), ("smile", "frown")
    ]
    q, a = random.choice(pairs)
    return f"what is the opposite of {q}?", a

def get_synonym():
    pairs = [
        ("happy", "joyful"), ("sad", "unhappy"), ("fast", "quick"), ("big", "large"),
        ("small", "tiny"), ("angry", "mad"), ("start", "begin"), ("end", "finish"),
        ("smart", "clever"), ("hard", "difficult"), ("easy", "simple"), ("scared", "afraid"),
        ("stone", "rock"), ("street", "road"), ("correct", "right"), ("incorrect", "wrong"),
        ("loud", "noisy"), ("quiet", "silent"), ("gift", "present"), ("talk", "speak"),
        ("listen", "hear"), ("jump", "leap"), ("run", "sprint"), ("look", "see"),
        ("choose", "pick"), ("help", "assist"), ("garbage", "trash")
    ]
    q, a = random.choice(pairs)
    return f"what is a synonym for {q}?", a

def get_sound():
    pairs = [
        ("dog", "woof"), ("cat", "meow"), ("cow", "moo"), ("duck", "quack"),
        ("lion", "roar"), ("pig", "oink"), ("bird", "chirp"), ("snake", "hiss"),
        ("sleeping person", "zzz"), ("bee", "buzz"), ("owl", "hoot"), ("sheep", "baa"),
        ("rooster", "cock-a-doodle-doo"), ("horse", "neigh"), ("mouse", "squeak"),
        ("frog", "ribbit"), ("wolf", "howl"), ("chicken", "cluck"), ("bell", "ding"),
        ("clock", "tick tock"), ("car", "vroom"), ("ghost", "boo"), ("train", "choo choo"),
        ("wind", "whoosh"), ("rain", "pitter patter"), ("thunder", "boom"),
        ("drum", "bang"), ("door", "knock"), ("camera", "click")
    ]
    animal, sound = random.choice(pairs)
    return f"what sound does a {animal} make?", sound

def get_rhyme():
    pairs = [
        ("cat", "bat"), ("sun", "run"), ("blue", "glue"), ("fish", "dish"),
        ("ball", "tall"), ("mouse", "house"), ("star", "car"), ("tree", "bee"), 
        ("truck", "luck"), ("tummy", "yummy"), ("lame", "game"), ("kill", "bill"),
        ("hate", "gate"), ("seeker", "beeker"), ("mice", "lice"), ("master", "disaster"),
        ("skill", "pill"), ("sound", "underground"), ("cucumber", "lumber"),
        ("river", "shiver"), ("zig", "zag"), ("spoil", "boil"), ("feed", "breed"),
        ("blood", "flood"), ("pump", "bump"), ("hesitate", "detonate"), ("shoes", "clues"),
        ("frog", "log"), ("moon", "spoon"), ("top", "stop"), ("man", "can"),
        ("dog", "fog"), ("lip", "ship"), ("red", "bed"), ("pen", "hen"),
        ("fox", "box"), ("bug", "mug"), ("night", "light"), ("ring", "king"),
        ("cake", "snake"), ("goat", "boat"), ("cry", "fly"), ("stair", "chair"),
        ("jar", "car"), ("map", "cap"), ("sky", "fly"), ("book", "cook"),
        ("wall", "call"), ("day", "play"), ("green", "seen")
    ]
    word, rhyme = random.choice(pairs)
    return f"what rhymes with {word}?", rhyme

def get_memory():
    items = [
        "key", "sword", "map", "apple", "book", "coin", "orange", "computer", 
        "stick", "broom", "bucket", "dog", "cat", "hat", "shoe", "pencil", 
        "cup", "sandwich", "guitar", "flower", "ring", "camera", "pillow",
        "blanket", "cookie", "bottle", "chair", "lamp", "hammer", "nail",
        "telephone", "wallet", "watch", "glasses", "backpack", "umbrella",
        "ticket", "passport", "radio", "compass"
    ]
    item = random.choice(items)
    return f"larry has a {item}. what does larry have?", item

def get_capitals():
    pairs = [
        ("usa", "washington dc"), ("france", "paris"), ("germany", "berlin"),
        ("italy", "rome"), ("spain", "madrid"), ("japan", "tokyo"),
        ("china", "beijing"), ("russia", "moscow"), ("uk", "london"),
        ("canada", "ottawa"), ("australia", "canberra"), ("india", "new delhi"),
        ("egypt", "cairo"), ("mexico", "mexico city"), ("brazil", "brasilia"),
        ("argentina", "buenos aires"), ("chile", "santiago"), ("peru", "lima"),
        ("thailand", "bangkok"), ("vietnam", "hanoi"), ("south korea", "seoul"),
        ("greece", "athens"), ("turkey", "ankara"), ("sweden", "stockholm"),
        ("norway", "oslo"), ("finland", "helsinki"), ("ireland", "dublin"),
        ("portugal", "lisbon"), ("netherlands", "amsterdam"), ("belgium", "brussels"),
        ("austria", "vienna"), ("switzerland", "bern"), ("poland", "warsaw")
    ]
    country, city = random.choice(pairs)
    return f"what is the capital of {country}?", city



import random

def get_taxonomy():
    pairs = [
        # --- ORIGINAL CATEGORIES (EXPANDED) ---
        # Fruits
        ("apple", "fruit"), ("banana", "fruit"), ("pear", "fruit"), ("grape", "fruit"), ("orange", "fruit"),
        ("lemon", "fruit"), ("lime", "fruit"), ("strawberry", "fruit"), ("blueberry", "fruit"), ("raspberry", "fruit"),
        ("blackberry", "fruit"), ("kiwi", "fruit"), ("mango", "fruit"), ("pineapple", "fruit"), ("watermelon", "fruit"),
        ("cantaloupe", "fruit"), ("honeydew", "fruit"), ("peach", "fruit"), ("nectarine", "fruit"), ("plum", "fruit"),
        ("apricot", "fruit"), ("cherry", "fruit"), ("pomegranate", "fruit"), ("fig", "fruit"), ("date", "fruit"),
        ("papaya", "fruit"), ("guava", "fruit"), ("lychee", "fruit"), ("dragonfruit", "fruit"), ("passionfruit", "fruit"),
        ("persimmon", "fruit"), ("kumquat", "fruit"), ("cranberry", "fruit"), ("coconut", "fruit"), ("avocado", "fruit"),

        # Vegetables
        ("carrot", "vegetable"), ("broccoli", "vegetable"), ("cauliflower", "vegetable"), ("spinach", "vegetable"),
        ("kale", "vegetable"), ("lettuce", "vegetable"), ("cabbage", "vegetable"), ("brussels sprout", "vegetable"),
        ("peas", "vegetable"), ("green bean", "vegetable"), ("zucchini", "vegetable"), ("eggplant", "vegetable"),
        ("cucumber", "vegetable"), ("bell pepper", "vegetable"), ("jalapeno", "vegetable"), ("chili pepper", "vegetable"),
        ("corn", "vegetable"), ("potato", "vegetable"), ("sweet potato", "vegetable"), ("yam", "vegetable"),
        ("onion", "vegetable"), ("garlic", "vegetable"), ("ginger", "vegetable"), ("radish", "vegetable"),
        ("beet", "vegetable"), ("turnip", "vegetable"), ("parsnip", "vegetable"), ("asparagus", "vegetable"),
        ("celery", "vegetable"), ("mushroom", "vegetable"), ("pumpkin", "vegetable"), ("squash", "vegetable"),
        ("artichoke", "vegetable"), ("okra", "vegetable"), ("leek", "vegetable"), ("shallot", "vegetable"),

        # Animals (Mammals)
        ("dog", "animal"), ("cat", "animal"), ("elephant", "animal"), ("lion", "animal"), ("tiger", "animal"),
        ("bear", "animal"), ("wolf", "animal"), ("fox", "animal"), ("rabbit", "animal"), ("deer", "animal"),
        ("moose", "animal"), ("elk", "animal"), ("horse", "animal"), ("cow", "animal"), ("pig", "animal"),
        ("sheep", "animal"), ("goat", "animal"), ("donkey", "animal"), ("monkey", "animal"), ("gorilla", "animal"),
        ("chimpanzee", "animal"), ("orangutan", "animal"), ("giraffe", "animal"), ("zebra", "animal"), ("hippo", "animal"),
        ("rhino", "animal"), ("kangaroo", "animal"), ("koala", "animal"), ("panda", "animal"), ("sloth", "animal"),
        ("otter", "animal"), ("beaver", "animal"), ("raccoon", "animal"), ("skunk", "animal"), ("squirrel", "animal"),
        ("chipmunk", "animal"), ("mouse", "animal"), ("rat", "animal"), ("hamster", "animal"), ("guinea pig", "animal"),
        ("hedgehog", "animal"), ("bat", "animal"), ("mole", "animal"), ("camel", "animal"), ("llama", "animal"),
        ("alpaca", "animal"), ("buffalo", "animal"), ("bison", "animal"), ("hyena", "animal"), ("leopard", "animal"),
        ("cheetah", "animal"), ("jaguar", "animal"), ("cougar", "animal"), ("lynx", "animal"), ("bobcat", "animal"),

        # Birds
        ("eagle", "bird"), ("hawk", "bird"), ("falcon", "bird"), ("owl", "bird"), ("vulture", "bird"),
        ("sparrow", "bird"), ("robin", "bird"), ("blue jay", "bird"), ("cardinal", "bird"), ("crow", "bird"),
        ("raven", "bird"), ("pigeon", "bird"), ("dove", "bird"), ("parrot", "bird"), ("parakeet", "bird"),
        ("cockatoo", "bird"), ("macaw", "bird"), ("canary", "bird"), ("finch", "bird"), ("hummingbird", "bird"),
        ("woodpecker", "bird"), ("duck", "bird"), ("goose", "bird"), ("swan", "bird"), ("seagull", "bird"),
        ("pelican", "bird"), ("stork", "bird"), ("heron", "bird"), ("flamingo", "bird"), ("crane", "bird"),
        ("penguin", "bird"), ("ostrich", "bird"), ("emu", "bird"), ("kiwi", "bird"), ("peacock", "bird"),
        ("turkey", "bird"), ("chicken", "bird"), ("rooster", "bird"), ("hen", "bird"), ("quail", "bird"),

        # Fish & Marine Life
        ("salmon", "fish"), ("tuna", "fish"), ("trout", "fish"), ("bass", "fish"), ("cod", "fish"),
        ("herring", "fish"), ("sardine", "fish"), ("mackerel", "fish"), ("snapper", "fish"), ("grouper", "fish"),
        ("halibut", "fish"), ("flounder", "fish"), ("catfish", "fish"), ("carp", "fish"), ("goldfish", "fish"),
        ("betta", "fish"), ("guppy", "fish"), ("tetra", "fish"), ("shark", "fish"), ("whale", "fish"),
        ("dolphin", "fish"), ("porpoise", "fish"), ("orca", "fish"), ("seal", "fish"), ("sea lion", "fish"),
        ("walrus", "fish"), ("manatee", "fish"), ("stingray", "fish"), ("eel", "fish"), ("seahorse", "fish"),
        ("jellyfish", "fish"), ("octopus", "fish"), ("squid", "fish"), ("crab", "fish"), ("lobster", "fish"),
        ("shrimp", "fish"), ("prawn", "fish"), ("oyster", "fish"), ("clam", "fish"), ("mussel", "fish"),

        # Insects & Arachnids
        ("ant", "insect"), ("bee", "insect"), ("wasp", "insect"), ("hornet", "insect"), ("butterfly", "insect"),
        ("moth", "insect"), ("fly", "insect"), ("mosquito", "insect"), ("dragonfly", "insect"), ("damselfly", "insect"),
        ("beetle", "insect"), ("ladybug", "insect"), ("cockroach", "insect"), ("cricket", "insect"), ("grasshopper", "insect"),
        ("locust", "insect"), ("praying mantis", "insect"), ("stick insect", "insect"), ("termite", "insect"), ("flea", "insect"),
        ("spider", "insect"), ("tarantula", "insect"), ("scorpion", "insect"), ("tick", "insect"), ("mite", "insect"),
        ("centipede", "insect"), ("millipede", "insect"), ("worm", "insect"), ("snail", "insect"), ("slug", "insect"),

        # Vehicles
        ("car", "vehicle"), ("truck", "vehicle"), ("van", "vehicle"), ("suv", "vehicle"), ("bus", "vehicle"),
        ("motorcycle", "vehicle"), ("scooter", "vehicle"), ("bicycle", "vehicle"), ("tricycle", "vehicle"), ("unicycle", "vehicle"),
        ("skateboard", "vehicle"), ("rollerblades", "vehicle"), ("train", "vehicle"), ("subway", "vehicle"), ("tram", "vehicle"),
        ("trolley", "vehicle"), ("monorail", "vehicle"), ("airplane", "vehicle"), ("jet", "vehicle"), ("helicopter", "vehicle"),
        ("glider", "vehicle"), ("blimp", "vehicle"), ("hot air balloon", "vehicle"), ("rocket", "vehicle"), ("spaceship", "vehicle"),
        ("boat", "vehicle"), ("ship", "vehicle"), ("yacht", "vehicle"), ("sailboat", "vehicle"), ("canoe", "vehicle"),
        ("kayak", "vehicle"), ("raft", "vehicle"), ("ferry", "vehicle"), ("cruise ship", "vehicle"), ("submarine", "vehicle"),
        ("tractor", "vehicle"), ("bulldozer", "vehicle"), ("crane", "vehicle"), ("forklift", "vehicle"), ("ambulance", "vehicle"),
        ("firetruck", "vehicle"), ("police car", "vehicle"), ("taxi", "vehicle"), ("limousine", "vehicle"), ("golf cart", "vehicle"),

        # Tools
        ("hammer", "tool"), ("screwdriver", "tool"), ("wrench", "tool"), ("pliers", "tool"), ("saw", "tool"),
        ("drill", "tool"), ("tape measure", "tool"), ("level", "tool"), ("chisel", "tool"), ("file", "tool"),
        ("sander", "tool"), ("grinder", "tool"), ("router", "tool"), ("planer", "tool"), ("lathe", "tool"),
        ("axe", "tool"), ("hatchet", "tool"), ("mallet", "tool"), ("sledgehammer", "tool"), ("crowbar", "tool"),
        ("shovel", "tool"), ("rake", "tool"), ("hoe", "tool"), ("trowel", "tool"), ("spade", "tool"),
        ("pickaxe", "tool"), ("pitchfork", "tool"), ("wheelbarrow", "tool"), ("ladder", "tool"), ("paintbrush", "tool"),
        ("roller", "tool"), ("wrench", "tool"), ("allen key", "tool"), ("socket", "tool"), ("vise", "tool"),
        ("clamp", "tool"), ("soldering iron", "tool"), ("welder", "tool"), ("multimeter", "tool"), ("flashlight", "tool"),

        # Clothing & Accessories
        ("shirt", "clothing"), ("t-shirt", "clothing"), ("blouse", "clothing"), ("polo", "clothing"), ("sweater", "clothing"),
        ("sweatshirt", "clothing"), ("hoodie", "clothing"), ("jacket", "clothing"), ("coat", "clothing"), ("parka", "clothing"),
        ("vest", "clothing"), ("suit", "clothing"), ("blazer", "clothing"), ("tuxedo", "clothing"), ("dress", "clothing"),
        ("skirt", "clothing"), ("pants", "clothing"), ("jeans", "clothing"), ("trousers", "clothing"), ("slacks", "clothing"),
        ("shorts", "clothing"), ("leggings", "clothing"), ("tights", "clothing"), ("pajamas", "clothing"), ("robe", "clothing"),
        ("underwear", "clothing"), ("boxers", "clothing"), ("briefs", "clothing"), ("bra", "clothing"), ("socks", "clothing"),
        ("shoes", "clothing"), ("sneakers", "clothing"), ("boots", "clothing"), ("sandals", "clothing"), ("flip-flops", "clothing"),
        ("heels", "clothing"), ("loafers", "clothing"), ("slippers", "clothing"), ("hat", "clothing"), ("cap", "clothing"),
        ("beanie", "clothing"), ("gloves", "clothing"), ("mittens", "clothing"), ("scarf", "clothing"), ("tie", "clothing"),
        ("bowtie", "clothing"), ("belt", "clothing"), ("sunglasses", "clothing"), ("watch", "clothing"), ("jewelry", "clothing"),

        # Colors
        ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"), ("orange", "color"),
        ("purple", "color"), ("pink", "color"), ("brown", "color"), ("black", "color"), ("white", "color"),
        ("gray", "color"), ("violet", "color"), ("indigo", "color"), ("cyan", "color"), ("magenta", "color"),
        ("teal", "color"), ("turquoise", "color"), ("lavender", "color"), ("maroon", "color"), ("burgundy", "color"),
        ("navy", "color"), ("olive", "color"), ("lime", "color"), ("beige", "color"), ("tan", "color"),
        ("cream", "color"), ("ivory", "color"), ("gold", "color"), ("silver", "color"), ("bronze", "color"),
        ("copper", "color"), ("platinum", "color"), ("charcoal", "color"), ("coral", "color"), ("salmon", "color"),
        ("peach", "color"), ("mint", "color"), ("mustard", "color"), ("rust", "color"), ("fuchsia", "color"),

        # Shapes
        ("circle", "shape"), ("square", "shape"), ("triangle", "shape"), ("rectangle", "shape"), ("oval", "shape"),
        ("ellipse", "shape"), ("pentagon", "shape"), ("hexagon", "shape"), ("octagon", "shape"), ("decagon", "shape"),
        ("rhombus", "shape"), ("parallelogram", "shape"), ("trapezoid", "shape"), ("star", "shape"), ("heart", "shape"),
        ("crescent", "shape"), ("cross", "shape"), ("diamond", "shape"), ("sphere", "shape"), ("cube", "shape"),
        ("cone", "shape"), ("cylinder", "shape"), ("pyramid", "shape"), ("prism", "shape"), ("torus", "shape"),
        ("spiral", "shape"), ("helix", "shape"), ("line", "shape"), ("point", "shape"), ("plane", "shape"),

        # Instruments
        ("piano", "instrument"), ("guitar", "instrument"), ("violin", "instrument"), ("cello", "instrument"), ("viola", "instrument"),
        ("double bass", "instrument"), ("harp", "instrument"), ("flute", "instrument"), ("piccolo", "instrument"), ("clarinet", "instrument"),
        ("oboe", "instrument"), ("bassoon", "instrument"), ("saxophone", "instrument"), ("trumpet", "instrument"), ("trombone", "instrument"),
        ("tuba", "instrument"), ("french horn", "instrument"), ("drums", "instrument"), ("cymbals", "instrument"), ("snare", "instrument"),
        ("timpani", "instrument"), ("xylophone", "instrument"), ("marimba", "instrument"), ("vibraphone", "instrument"), ("glockenspiel", "instrument"),
        ("accordion", "instrument"), ("harmonica", "instrument"), ("bagpipes", "instrument"), ("banjo", "instrument"), ("ukulele", "instrument"),
        ("mandolin", "instrument"), ("sitar", "instrument"), ("lute", "instrument"), ("organ", "instrument"), ("synthesizer", "instrument"),
        ("keyboard", "instrument"), ("electric guitar", "instrument"), ("bass guitar", "instrument"), ("drum machine", "instrument"), ("turntable", "instrument"),

        # Flowers
        ("rose", "flower"), ("tulip", "flower"), ("daisy", "flower"), ("sunflower", "flower"), ("lily", "flower"),
        ("orchid", "flower"), ("carnation", "flower"), ("daffodil", "flower"), ("hyacinth", "flower"), ("iris", "flower"),
        ("peony", "flower"), ("chrysanthemum", "flower"), ("dahlia", "flower"), ("gerbera", "flower"), ("marigold", "flower"),
        ("begonia", "flower"), ("petunia", "flower"), ("pansy", "flower"), ("violet", "flower"), ("poppy", "flower"),
        ("snapdragon", "flower"), ("lavender", "flower"), ("lilac", "flower"), ("jasmine", "flower"), ("gardenia", "flower"),
        ("magnolia", "flower"), ("camellia", "flower"), ("azalea", "flower"), ("rhododendron", "flower"), ("hibiscus", "flower"),
        ("lotus", "flower"), ("water lily", "flower"), ("anemone", "flower"), ("ranunculus", "flower"), ("sweet pea", "flower"),

        # Trees
        ("oak", "tree"), ("maple", "tree"), ("pine", "tree"), ("spruce", "tree"), ("fir", "tree"),
        ("cedar", "tree"), ("redwood", "tree"), ("sequoia", "tree"), ("birch", "tree"), ("aspen", "tree"),
        ("willow", "tree"), ("poplar", "tree"), ("elm", "tree"), ("ash", "tree"), ("beech", "tree"),
        ("walnut", "tree"), ("hickory", "tree"), ("chestnut", "tree"), ("pecan", "tree"), ("sycamore", "tree"),
        ("locust", "tree"), ("magnolia", "tree"), ("dogwood", "tree"), ("cherry", "tree"), ("apple", "tree"),
        ("pear", "tree"), ("peach", "tree"), ("plum", "tree"), ("orange", "tree"), ("lemon", "tree"),
        ("palm", "tree"), ("coconut", "tree"), ("banyan", "tree"), ("baobab", "tree"), ("bamboo", "tree"),
        ("eucalyptus", "tree"), ("acacia", "tree"), ("mahogany", "tree"), ("teak", "tree"), ("ebony", "tree"),

        # Metals & Materials
        ("iron", "metal"), ("gold", "metal"), ("silver", "metal"), ("copper", "metal"), ("bronze", "metal"),
        ("brass", "metal"), ("steel", "metal"), ("aluminum", "metal"), ("titanium", "metal"), ("platinum", "metal"),
        ("lead", "metal"), ("tin", "metal"), ("zinc", "metal"), ("nickel", "metal"), ("chrome", "metal"),
        ("mercury", "metal"), ("tungsten", "metal"), ("magnesium", "metal"), ("cobalt", "metal"), ("uranium", "metal"),
        ("wood", "material"), ("glass", "material"), ("plastic", "material"), ("rubber", "material"), ("stone", "material"),
        ("brick", "material"), ("concrete", "material"), ("ceramic", "material"), ("porcelain", "material"), ("clay", "material"),
        ("leather", "material"), ("fur", "material"), ("wool", "material"), ("cotton", "material"), ("silk", "material"),
        ("linen", "material"), ("polyester", "material"), ("nylon", "material"), ("spandex", "material"), ("velvet", "material"),

        # Sports
        ("soccer", "sport"), ("basketball", "sport"), ("football", "sport"), ("baseball", "sport"), ("tennis", "sport"),
        ("volleyball", "sport"), ("hockey", "sport"), ("golf", "sport"), ("rugby", "sport"), ("cricket", "sport"),
        ("boxing", "sport"), ("wrestling", "sport"), ("mma", "sport"), ("judo", "sport"), ("karate", "sport"),
        ("taekwondo", "sport"), ("kung fu", "sport"), ("fencing", "sport"), ("archery", "sport"), ("shooting", "sport"),
        ("swimming", "sport"), ("diving", "sport"), ("surfing", "sport"), ("water polo", "sport"), ("rowing", "sport"),
        ("canoeing", "sport"), ("kayaking", "sport"), ("sailing", "sport"), ("skiing", "sport"), ("snowboarding", "sport"),
        ("skating", "sport"), ("ice skating", "sport"), ("roller skating", "sport"), ("skateboarding", "sport"), ("cycling", "sport"),
        ("running", "sport"), ("sprinting", "sport"), ("marathon", "sport"), ("triathlon", "sport"), ("gymnastics", "sport"),

        # Drinks
        ("water", "drink"), ("milk", "drink"), ("juice", "drink"), ("soda", "drink"), ("tea", "drink"),
        ("coffee", "drink"), ("espresso", "drink"), ("latte", "drink"), ("cappuccino", "drink"), ("mocha", "drink"),
        ("hot chocolate", "drink"), ("lemonade", "drink"), ("iced tea", "drink"), ("smoothie", "drink"), ("milkshake", "drink"),
        ("beer", "drink"), ("wine", "drink"), ("whiskey", "drink"), ("vodka", "drink"), ("rum", "drink"),
        ("gin", "drink"), ("tequila", "drink"), ("brandy", "drink"), ("cider", "drink"), ("champagne", "drink"),
        ("cocktail", "drink"), ("margarita", "drink"), ("martini", "drink"), ("mojito", "drink"), ("mimosa", "drink"),
        ("root beer", "drink"), ("ginger ale", "drink"), ("cola", "drink"), ("tonic water", "drink"), ("club soda", "drink"),

        # Food (Prepared)
        ("bread", "food"), ("toast", "food"), ("sandwich", "food"), ("burger", "food"), ("hot dog", "food"),
        ("pizza", "food"), ("pasta", "food"), ("spaghetti", "food"), ("lasagna", "food"), ("ravioli", "food"),
        ("macaroni", "food"), ("noodle", "food"), ("ramen", "food"), ("rice", "food"), ("sushi", "food"),
        ("sashimi", "food"), ("taco", "food"), ("burrito", "food"), ("quesadilla", "food"), ("enchilada", "food"),
        ("nachos", "food"), ("steak", "food"), ("chicken", "food"), ("roast beef", "food"), ("pork chop", "food"),
        ("ribs", "food"), ("sausage", "food"), ("bacon", "food"), ("ham", "food"), ("egg", "food"),
        ("omelet", "food"), ("pancake", "food"), ("waffle", "food"), ("french toast", "food"), ("cereal", "food"),
        ("oatmeal", "food"), ("yogurt", "food"), ("cheese", "food"), ("butter", "food"), ("cream", "food"),
        ("soup", "food"), ("stew", "food"), ("chili", "food"), ("salad", "food"), ("curry", "food"),
        ("pie", "food"), ("cake", "food"), ("cookie", "food"), ("brownie", "food"), ("donut", "food"),
        ("ice cream", "food"), ("chocolate", "food"), ("candy", "food"), ("popcorn", "food"), ("chips", "food"),

        # --- NEW CATEGORIES ---
        # Furniture
        ("chair", "furniture"), ("table", "furniture"), ("desk", "furniture"), ("sofa", "furniture"), ("couch", "furniture"),
        ("bed", "furniture"), ("mattress", "furniture"), ("dresser", "furniture"), ("cabinet", "furniture"), ("bookshelf", "furniture"),
        ("shelf", "furniture"), ("nightstand", "furniture"), ("coffee table", "furniture"), ("dining table", "furniture"), ("bench", "furniture"),
        ("stool", "furniture"), ("ottoman", "furniture"), ("armchair", "furniture"), ("recliner", "furniture"), ("futon", "furniture"),
        ("wardrobe", "furniture"), ("closet", "furniture"), ("cupboard", "furniture"), ("pantry", "furniture"), ("credenza", "furniture"),
        ("vanity", "furniture"), ("crib", "furniture"), ("cradle", "furniture"), ("high chair", "furniture"), ("bunk bed", "furniture"),

        # Electronics
        ("computer", "electronics"), ("laptop", "electronics"), ("desktop", "electronics"), ("tablet", "electronics"), ("smartphone", "electronics"),
        ("phone", "electronics"), ("watch", "electronics"), ("smartwatch", "electronics"), ("television", "electronics"), ("tv", "electronics"),
        ("monitor", "electronics"), ("screen", "electronics"), ("projector", "electronics"), ("camera", "electronics"), ("camcorder", "electronics"),
        ("webcam", "electronics"), ("microphone", "electronics"), ("speaker", "electronics"), ("headphones", "electronics"), ("earbuds", "electronics"),
        ("headset", "electronics"), ("keyboard", "electronics"), ("mouse", "electronics"), ("trackpad", "electronics"), ("printer", "electronics"),
        ("scanner", "electronics"), ("router", "electronics"), ("modem", "electronics"), ("hard drive", "electronics"), ("usb drive", "electronics"),
        ("console", "electronics"), ("controller", "electronics"), ("remote", "electronics"), ("battery", "electronics"), ("charger", "electronics"),
        ("cable", "electronics"), ("adapter", "electronics"), ("server", "electronics"), ("drone", "electronics"), ("robot", "electronics"),

        # Professions
        ("doctor", "profession"), ("nurse", "profession"), ("surgeon", "profession"), ("dentist", "profession"), ("pharmacist", "profession"),
        ("teacher", "profession"), ("professor", "profession"), ("student", "profession"), ("principal", "profession"), ("librarian", "profession"),
        ("engineer", "profession"), ("scientist", "profession"), ("researcher", "profession"), ("architect", "profession"), ("designer", "profession"),
        ("artist", "profession"), ("painter", "profession"), ("sculptor", "profession"), ("writer", "profession"), ("author", "profession"),
        ("journalist", "profession"), ("editor", "profession"), ("photographer", "profession"), ("musician", "profession"), ("singer", "profession"),
        ("actor", "profession"), ("actress", "profession"), ("director", "profession"), ("producer", "profession"), ("dancer", "profession"),
        ("chef", "profession"), ("cook", "profession"), ("baker", "profession"), ("waiter", "profession"), ("waitress", "profession"),
        ("bartender", "profession"), ("pilot", "profession"), ("flight attendant", "profession"), ("captain", "profession"), ("driver", "profession"),
        ("police officer", "profession"), ("detective", "profession"), ("firefighter", "profession"), ("paramedic", "profession"), ("soldier", "profession"),
        ("lawyer", "profession"), ("judge", "profession"), ("attorney", "profession"), ("politician", "profession"), ("accountant", "profession"),
        ("manager", "profession"), ("ceo", "profession"), ("secretary", "profession"), ("receptionist", "profession"), ("clerk", "profession"),
        ("electrician", "profession"), ("plumber", "profession"), ("carpenter", "profession"), ("mechanic", "profession"), ("farmer", "profession"),

        # Countries
        ("usa", "country"), ("canada", "country"), ("mexico", "country"), ("brazil", "country"), ("argentina", "country"),
        ("chile", "country"), ("peru", "country"), ("colombia", "country"), ("venezuela", "country"), ("uk", "country"),
        ("france", "country"), ("germany", "country"), ("italy", "country"), ("spain", "country"), ("portugal", "country"),
        ("netherlands", "country"), ("belgium", "country"), ("switzerland", "country"), ("austria", "country"), ("sweden", "country"),
        ("norway", "country"), ("denmark", "country"), ("finland", "country"), ("ireland", "country"), ("poland", "country"),
        ("russia", "country"), ("ukraine", "country"), ("greece", "country"), ("turkey", "country"), ("egypt", "country"),
        ("morocco", "country"), ("nigeria", "country"), ("south africa", "country"), ("kenya", "country"), ("ethiopia", "country"),
        ("china", "country"), ("japan", "country"), ("india", "country"), ("south korea", "country"), ("north korea", "country"),
        ("vietnam", "country"), ("thailand", "country"), ("indonesia", "country"), ("malaysia", "country"), ("philippines", "country"),
        ("australia", "country"), ("new zealand", "country"), ("fiji", "country"), ("israel", "country"), ("saudi arabia", "country"),
        ("iran", "country"), ("iraq", "country"), ("pakistan", "country"), ("afghanistan", "country"), ("bangladesh", "country"),

        # Cities (Major)
        ("new york", "city"), ("los angeles", "city"), ("chicago", "city"), ("houston", "city"), ("phoenix", "city"),
        ("london", "city"), ("paris", "city"), ("berlin", "city"), ("rome", "city"), ("madrid", "city"),
        ("moscow", "city"), ("tokyo", "city"), ("beijing", "city"), ("shanghai", "city"), ("seoul", "city"),
        ("bangkok", "city"), ("mumbai", "city"), ("delhi", "city"), ("cairo", "city"), ("istanbul", "city"),
        ("dubai", "city"), ("toronto", "city"), ("mexico city", "city"), ("sao paulo", "city"), ("buenos aires", "city"),
        ("sydney", "city"), ("melbourne", "city"), ("lagos", "city"), ("jakarta", "city"), ("karachi", "city"),

        # Space
        ("mercury", "planet"), ("venus", "planet"), ("earth", "planet"), ("mars", "planet"), ("jupiter", "planet"),
        ("saturn", "planet"), ("uranus", "planet"), ("neptune", "planet"), ("pluto", "dwarf planet"), ("sun", "star"),
        ("moon", "moon"), ("titan", "moon"), ("europa", "moon"), ("ganymede", "moon"), ("io", "moon"),
        ("callisto", "moon"), ("phobos", "moon"), ("deimos", "moon"), ("asteroid", "space object"), ("comet", "space object"),
        ("meteor", "space object"), ("galaxy", "space object"), ("nebula", "space object"), ("black hole", "space object"), ("star", "space object"),
        ("constellation", "space object"), ("supernova", "space object"), ("milky way", "galaxy"), ("andromeda", "galaxy"), ("universe", "space object"),

        # Elements (Chemistry)
        ("hydrogen", "element"), ("helium", "element"), ("lithium", "element"), ("beryllium", "element"), ("boron", "element"),
        ("carbon", "element"), ("nitrogen", "element"), ("oxygen", "element"), ("fluorine", "element"), ("neon", "element"),
        ("sodium", "element"), ("magnesium", "element"), ("aluminum", "element"), ("silicon", "element"), ("phosphorus", "element"),
        ("sulfur", "element"), ("chlorine", "element"), ("argon", "element"), ("potassium", "element"), ("calcium", "element"),
        ("scandium", "element"), ("titanium", "element"), ("vanadium", "element"), ("chromium", "element"), ("manganese", "element"),
        ("iron", "element"), ("cobalt", "element"), ("nickel", "element"), ("copper", "element"), ("zinc", "element"),
        ("gallium", "element"), ("germanium", "element"), ("arsenic", "element"), ("selenium", "element"), ("bromine", "element"),
        ("krypton", "element"), ("rubidium", "element"), ("strontium", "element"), ("yttrium", "element"), ("zirconium", "element"),

        # Body Parts
        ("head", "body part"), ("hair", "body part"), ("face", "body part"), ("eye", "body part"), ("ear", "body part"),
        ("nose", "body part"), ("mouth", "body part"), ("lip", "body part"), ("tooth", "body part"), ("tongue", "body part"),
        ("neck", "body part"), ("shoulder", "body part"), ("arm", "body part"), ("elbow", "body part"), ("wrist", "body part"),
        ("hand", "body part"), ("finger", "body part"), ("thumb", "body part"), ("chest", "body part"), ("stomach", "body part"),
        ("back", "body part"), ("spine", "body part"), ("hip", "body part"), ("leg", "body part"), ("knee", "body part"),
        ("ankle", "body part"), ("foot", "body part"), ("toe", "body part"), ("skin", "body part"), ("bone", "body part"),
    ]
    item, category = random.choice(pairs)
    if random.random() > 0.5:
        return f"what category is {item}?", category
    else:
        return f"is {item} a {category}?", "yes"

def get_unscramble():
    words = [
        "cat", "dog", "bat", "rat", "hat", "map", "box", "fox", "pig", "sun",
        "run", "fun", "sky", "fly", "pie", "key", "pen", "cup", "car", "bus",
        "bed", "red", "one", "two", "six", "ten", "apple", "fruit", "banana", 
        "carrot", "vegetable", "salmon", "fish", "eagle", "ant", "insect",
        "car", "vehicle", "pants", "bus", "clothing", "blue", "color", "flower",
        "piano", "drum", "rose", "oak", "iron", "metal", "gold", "silver", "soccer",
        "sport", "tennis", "football", "baseball", "milk", "bread", "water", "juice",
        "drink", "austria", "lisbon", "bern", "belgium", "telephone", "sword", "sandwitch",
        "guitar", "flower", "ring", "camera", "pillow", "blanket", "cookie", "juice", "bottle",
        "peanut", "butter", "chocolate", "sprite", "drink", "computer", "watch", "wallet", "glasses",
        "backpack", "umbrella", "ticket", "passport", "password", "radio", "compass",
    ]
    word = random.choice(words)
    chars = list(word)
    random.shuffle(chars)
    scrambled = "".join(chars)
    
    while scrambled == word:
        random.shuffle(chars)
        scrambled = "".join(chars)
        
    return f"unscramble the word {scrambled}", word

def get_colors():
    mixes = [
        ("red", "blue", "purple"),
        ("blue", "yellow", "green"),
        ("yellow", "red", "orange"),
        ("black", "white", "gray"),
        ("red", "white", "pink"),
        ("blue", "white", "light blue"),
        ("black", "red", "dark red")
    ]
    c1, c2, result = random.choice(mixes)
    return f"what do you get if you mix {c1} and {c2}?", result

def get_shapes():
    # NEW CATEGORY: Geometry basics
    shapes = [
        ("triangle", "3"), ("square", "4"), ("rectangle", "4"), ("pentagon", "5"),
        ("hexagon", "6"), ("octagon", "8"), ("circle", "0"), ("oval", "0")
    ]
    shape, sides = random.choice(shapes)
    return f"how many sides does a {shape} have?", sides

def get_coin_math():
    # NEW CATEGORY: Money math
    coins = {"penny": 1, "nickel": 5, "dime": 10, "quarter": 25}
    c1_name = random.choice(list(coins.keys()))
    c2_name = random.choice(list(coins.keys()))
    
    val1 = coins[c1_name]
    val2 = coins[c2_name]
    total = val1 + val2
    
    return f"what is the value of one {c1_name} and one {c2_name}?", str(total)

def get_time_conversion():
    # NEW CATEGORY: Time units
    choice = random.randint(1, 3)
    if choice == 1:
        # Hours to minutes
        hours = random.randint(1, 10)
        return f"how many minutes are in {hours} hours?", str(hours * 60)
    elif choice == 2:
        # Days to hours
        days = random.randint(1, 5)
        return f"how many hours are in {days} days?", str(days * 24)
    else:
        # Weeks to days
        weeks = random.randint(1, 10)
        return f"how many days are in {weeks} weeks?", str(weeks * 7)

def get_translation():
    # NEW CATEGORY: Basic Spanish
    pairs = [
        ("dog", "perro"), ("cat", "gato"), ("house", "casa"), ("water", "agua"),
        ("friend", "amigo"), ("red", "rojo"), ("blue", "azul"), ("one", "uno"),
        ("two", "dos"), ("milk", "leche"), ("book", "libro"), ("sun", "sol")
    ]
    eng, esp = random.choice(pairs)
    return f"translate {eng} to spanish", esp

def get_spelling():
    # NEW CATEGORY: Spelling
    words = ["cat", "dog", "fish", "bird", "book", "desk", "tree", "moon", "star", "hand"]
    word = random.choice(words)
    spelling = "-".join(list(word))
    return f"spell the word {word}", spelling

def get_general():
    qs = [
        # --- Identity & Basic Facts ---
        ("who are you?", "i am larry."),
        ("what is your name?", "larry"),
        ("what is your job?", "i answer questions."),
        ("are you a human?", "no"),
        ("are you a computer?", "yes"),
        ("what do you eat?", "electricity"),
        ("do you sleep?", "no"),
        
        # --- Colors & Nature ---
        ("what is the color of the sky?", "blue"),
        ("what is the color of grass?", "green"),
        ("what is the color of the sun?", "yellow"),
        ("what is the color of a banana?", "yellow"),
        ("what is the color of an apple?", "red"),
        ("what is the color of a carrot?", "orange"),
        ("what is the color of snow?", "white"),
        ("what is the color of coal?", "black"),
        ("what is the color of blood?", "red"),
        ("what is the color of a leaf?", "green"),
        ("is the sky green?", "no"),
        ("is grass red?", "no"),
        
        # --- Physics & Properties ---
        ("is fire hot?", "yes"),
        ("is ice hot?", "no"),
        ("is ice cold?", "yes"),
        ("is fire cold?", "no"),
        ("is water wet?", "yes"),
        ("is rock hard?", "yes"),
        ("is a pillow soft?", "yes"),
        ("is a feather heavy?", "no"),
        ("is a rock heavy?", "yes"),
        ("does wood float?", "yes"),
        ("does stone float?", "no"),
        ("can you see air?", "no"),
        ("is sugar sweet?", "yes"),
        ("is lemon sour?", "yes"),
        
        # --- Time & Numbers ---
        ("how many days in a week?", "7"),
        ("how many hours in a day?", "24"),
        ("how many seconds in a minute?", "60"),
        ("how many minutes in an hour?", "60"),
        ("how many months in a year?", "12"),
        ("what comes after monday?", "tuesday"),
        ("what comes before tuesday?", "monday"),
        ("what comes after friday?", "saturday"),
        
        # --- Biology & Animals ---
        ("how many legs does a dog have?", "4"),
        ("how many legs does a spider have?", "8"),
        ("how many legs does a bird have?", "2"),
        ("do fish swim?", "yes"),
        ("do birds fly?", "yes"),
        ("do dogs fly?", "no"),
        ("do snakes walk?", "no"),
        ("where do fish live?", "water"),
        ("what do cows drink?", "water"),
        ("what do bees make?", "honey"),
        ("do cats bark?", "no"),
        
        # --- Body Parts & Senses ---
        ("what do you use to see?", "eyes"),
        ("what do you use to hear?", "ears"),
        ("what do you use to smell?", "nose"),
        ("what do you use to walk?", "legs"),
        ("what do you use to hold things?", "hands"),
        ("how many fingers on one hand?", "5"),
        ("how many toes on one foot?", "5"),
        ("do you have eyes?", "no"),
        
        # --- Objects & Functions ---
        ("what do you wear on your feet?", "shoes"),
        ("what do you wear on your head?", "hat"),
        ("what do you use to write?", "pencil"),
        ("what do you use to cut paper?", "scissors"),
        ("what do you use to brush teeth?", "toothbrush"),
        ("what do you sleep on?", "bed"),
        ("what do you sit on?", "chair"),
        ("what do you drink from?", "cup"),
        ("does a car have wheels?", "yes"),
        ("how many wheels does a car have?", "4"),
        ("how many wheels does a bike have?", "2"),
        ("what do you use to call people?", "phone"),
        
        # --- World & Space ---
        ("what planet do we live on?", "earth"),
        ("what gives us light in the day?", "sun"),
        ("what gives us light at night?", "moon"),
        ("what direction does the sun rise?", "east"),
        ("what direction does the sun set?", "west"),
        ("is the ocean deep?", "yes"),
        ("is the desert dry?", "yes"),
        ("does it rain water?", "yes")
    ]
    return random.choice(qs)

def generate_unique_data(count):
    data = []
    choice = 0 
    attempts = 0
    max_attempts = count * 15  # Increased cushion for safety
    
    while len(data) < count:
        
        # INCREASED RANGE: Now randomizing between 1 and 18
        choice = random.randint(1, 18)

        if choice == 1: q, a = get_math()
        elif choice == 2: q, a = get_sequence()
        elif choice == 3: q, a = get_antonym()
        elif choice == 4: q, a = get_sound()
        elif choice == 5: q, a = get_rhyme()
        elif choice == 6: q, a = get_memory()
        elif choice == 7: q, a = get_general()
        elif choice == 8: q, a = get_capitals() 
        elif choice == 9: q, a = get_taxonomy()
        elif choice == 10: q, a = get_synonym() 
        elif choice == 11: q, a = get_comparison()
        elif choice == 12: q, a = get_unscramble()
        elif choice == 13: q, a = get_colors()
        elif choice == 14: q, a = get_shapes()         
        elif choice == 15: q, a = get_coin_math()     
        elif choice == 16: q, a = get_time_conversion() 
        elif choice == 17: q, a = get_translation()    
        elif choice == 18: q, a = get_spelling()       
        
        entry_signature = f"{q}|||{a}"
 
        if entry_signature not in seen_pairs:
            seen_pairs.add(entry_signature)
            data.append(f"{q}\n{a}\n")
        
    if len(data) < count:
        print(f"Warning: Only could generate {len(data)} unique items out of {count} requested.")
    
    random.shuffle(data)
    return data

def generate_data(count):
    data = []
    choice = 0 
    attempts = 0
    
    while len(data) < count:
        
        # INCREASED RANGE: Now randomizing between 1 and 18
        choice = random.randint(1, 18)

        if choice == 1: q, a = get_math()
        elif choice == 2: q, a = get_sequence()
        elif choice == 3: q, a = get_antonym()
        elif choice == 4: q, a = get_sound()
        elif choice == 5: q, a = get_rhyme()
        elif choice == 6: q, a = get_memory()
        elif choice == 7: q, a = get_general()
        elif choice == 8: q, a = get_capitals() 
        elif choice == 9: q, a = get_taxonomy()
        elif choice == 10: q, a = get_synonym() 
        elif choice == 11: q, a = get_comparison()
        elif choice == 12: q, a = get_unscramble()
        elif choice == 13: q, a = get_colors()
        elif choice == 14: q, a = get_shapes()         
        elif choice == 15: q, a = get_coin_math()     
        elif choice == 16: q, a = get_time_conversion() 
        elif choice == 17: q, a = get_translation()    
        elif choice == 18: q, a = get_spelling()       
        
        entry_signature = f"{q}|||{a}"
 
        if entry_signature not in seen_pairs:
            seen_pairs.add(entry_signature)
            
        data.append(f"{q}\n{a}\n")
        
    if len(data) < count:
        print(f"Warning: Only could generate {len(data)} unique items out of {count} requested.")
    
    random.shuffle(data)
    return data


def write_file(filename, data):
    with open(filename, 'w') as f:
        f.writelines(data)

# Main Execution
print(f"Generating {TRAIN_LINES} advanced training pairs...")
train_data = generate_unique_data(TRAIN_LINES)
write_file("corpus_clean.txt", train_data)
#
#print(f"Generating {VALID_LINES} advanced training pairs...")
#train_data = generate_unique_data(VALID_LINES)
#write_file("valid.txt", train_data)
#
print("Done.")
