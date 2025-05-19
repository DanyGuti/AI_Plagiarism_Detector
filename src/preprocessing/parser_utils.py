'''
This module provides utility functions for parsing Java code using ANTLR.'''
from antlr4 import *
from antlr.Java20Lexer import Java20Lexer
from antlr.Java20Parser import Java20Parser
from antlr.JavaVisitorImpl import JavaParserVisitorImpl
from antlr.Java20ParserVisitor import Java20ParserVisitor

def parse_java_code(code: str):
    """
    Parse Java code using ANTLR and return the parse tree.
    """
    input_stream = InputStream(code)
    lexer = Java20Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = Java20Parser(stream)
    tree = parser.compilationUnit()
    # Class to override methods to create
    # graph representation of the AST
    # and extract features (use the next two declarations for pre-processing)
    visitor = JavaParserVisitorImpl()
    return visitor.visit(tree)
