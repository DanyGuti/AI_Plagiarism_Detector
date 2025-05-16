'''
This module provides utility functions for parsing Java code using ANTLR.'''
from antlr4 import *
from antlr.JavaLexer import JavaLexer
from antlr.JavaParser import JavaParser
from antlr.JavaVisitorImpl import JavaParserVisitorImpl

def parse_java_code(code: str):
    """
    Parse Java code using ANTLR and return the parse tree.
    """
    input_stream = InputStream(code)
    lexer = JavaLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    tree = parser.compilationUnit()
    visitor = JavaParserVisitorImpl()
    return visitor.visit(tree)
