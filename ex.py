PRE_REQS = { # Courses not listed here have no pre-requisites
"Math 232": ["Math 231"], "Math 231": ["Math 112"],
"CS 212": ["CS 211"], "CS 211": ["CS 210"], "CS 210": ["Math 112"],
"CS 313": ["Math 232", "CS 212"],
"DSCI 102": ["DSCI 101", "Math 101"] }


def requires(course: str, other: str) -> bool:
    """Does taking one course require taking other course?
    >>> requires("Math 232", "Math 112")
    True
    >>> requires("DSCI 102", "Math 112")
    False
    """

    if not course in PRE_REQS:
        return course == other or False
    ever_found      = False 

    dependencies    = PRE_REQS[course]

    for dependency in dependencies:
        if other == dependency:
            ever_found = True 
        ever_found = ever_found or requires(dependency,other)

    return ever_found 

     



print(requires("CS 313", "Math 112"))
print(requires("DSCI 102", "CS 313"))
