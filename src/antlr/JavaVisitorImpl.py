from antlr4 import *
from antlr.JavaParserListener import JavaParserListener

class JavaParserVisitorImpl(JavaParserListener):
    def visitImportDeclaration(self, ctx):
        '''
        Handle import declarations in Java code.
        '''
        # Handle import declarations
        print("Import Declaration:", ctx.getText())
        return self.visitChildren(ctx)

    def visitStatement(self, ctx):
        '''
        Handle statements in the Java code.
        This includes variable declarations, method calls, etc.
        '''
        try:
            text = ctx.getText()
            if text:
                print("Statement:", text)
        except Exception as e:
            print(f"Error processing statement: {e}")
        return self.visitChildren(ctx)

    def visitMethodDeclaration(self, ctx):
        '''
        Handle method declarations.
        '''
        return {
            "type": "MethodDeclaration",
            "name": ctx.typeType().getText() if ctx.typeType() else "unknown",
            "children": self.visitChildren(ctx)
        }

    def visitClassDeclaration(self, ctx):
        '''
        Handle class declarations.
        '''
        return {
            "type": "ClassDeclaration",
            "name": ctx.IDENTIFIER().getText() if ctx.IDENTIFIER() else "unknown",
            "children": self.visitChildren(ctx)
        }

    def visitCompilationUnit(self, ctx):
        '''
        Handle the compilation unit (the entire Java file).
        '''
        return {
            "type": "CompilationUnit",
            "children": self.visitChildren(ctx)
        }

    def visitChildren(self, node):
        '''
        Visit all children of a node and return their results.
        '''
        results: list = []
        for child in node.getChildren():
            result = child.accept(self)
            if result is not None:
                results.append(result)
        return results
