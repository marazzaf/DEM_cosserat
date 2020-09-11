#coding: utf-8

class DEMError(Exception):
    """General class to raise an error in the process of doing a DEM computation."""
    pass

class ConvexError(DEMError):
    """Raises an error in case the algorithm cannot find a convex containing the barycenter of the facet.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message): #expression, message):
        #self.expression = expression
        self.message = message

