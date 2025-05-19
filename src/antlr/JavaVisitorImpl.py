from antlr4 import *
from .Java20ParserVisitor import Java20ParserVisitor as JParserVisitor
from .Java20Parser import Java20Parser

class JavaParserVisitorImpl(JParserVisitor):
    node_features : dict[int, dict[str, str | list[int]]] = {}
    def __init__(self):
        super().__init__()
        # monotonic increasing id for each node
        self._next_id = 0
        # mapping from node id to node features
        self.node_features : dict[int, dict[str, str | list[int]]] = {}
        # adjacency list: parent -> children
        self.edges : dict[int,list[int]] = {}

    def visitImportDeclaration(self, ctx):
        '''
        Handle import declarations in Java code.
        '''
       
        return None

    def visitStatement(self, ctx):
        '''
        Handle statements in the Java code.
        This includes variable declarations, method calls, etc.
        '''
        try:
            text = ctx.getText()
         
        except Exception as e:
            print(f"Error processing statement: {e}")
        return {
            "type": "Statement",
            "text": ctx.getText(),
            "children": self.visitChildren(ctx)
        }

    def visitMethodDeclaration(self, ctx):
        '''
        Handle method declarations.
        '''
        header_ctx = ctx.methodHeader()
        id_node = ctx.getToken(Java20Parser.Identifier, 0)
        method_name = id_node.getText() if id_node else None
        
        result_ctx = header_ctx.getTypedRuleContext(Java20Parser.ResultContext, 0)
        # Return type
        return_type = result_ctx.getText() if result_ctx else None
        return {
            "type": "MethodDeclaration",
            "name": method_name,
            "returnType": return_type,
            "children": self.visitChildren(ctx)
        }

    def visitClassDeclaration(self, ctx):
        '''
        Handle class declarations.
        '''
        id_node = ctx.getToken(Java20Parser.Identifier, 0)
        class_name = id_node.getText() if id_node else None
        return {
            "type": "ClassDeclaration",
            "name": class_name,
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
    def visitTerminal(self, node):
        return {
            "type": "Terminal",
            "text": node.getText()
        }

    def visitChildren(self, node):
        '''
        Visit all children of a node and return their results.
        '''
        results: list = []
        for child in node.getChildren():
            if isinstance(child, TerminalNode):
                results.append({
                    "type": "Terminal",
                    "text": child.getText()
                })
            else:
                ast = child.accept(self)
                if ast is not None:
                    results.append(ast)
        if hasattr(node, "getText"):
            rule_index = node.getRuleIndex()
            rule_name = Java20Parser.ruleNames[rule_index]
            return {    
                "type": rule_name,
                "text": node.getText(),
                "children": results
            }
        return results
