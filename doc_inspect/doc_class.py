
class Inspector:
    """This is a class for holding the basic data created when
    when reading and parsing a text. It holds all of the the basic
    attributes that the other classes work off of:

    Args:
        path (List[String): List of paths for input
         new (Boolean): Determines whether or not new data is being\
                       created or whether old data is being loaded

    Attributes:
        paths (List[String]): List of paths for input
        new (Boolean): Determines whether or not new data is being\
                       created or whether old data is being loaded

    """
    def __init__(self, path, new):
        self.paths = path

    def summary(self):
        """Returns corpora name"""
        return str(self.paths)
