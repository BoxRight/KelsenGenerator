#include <jansson.h>
#include "symbol_table.h"
#include "ast.h"

SymbolTable *symbol_table = NULL;  // Allocate and initialize the symbol table

json_t *ast_to_json(ASTNode *node, int level) {
    if (!node) return NULL;

    json_t *json_node = json_object();
    json_object_set_new(json_node, "NodeType", json_string(node_type_to_string(node->type)));
    json_object_set_new(json_node, "Level", json_integer(level));

    switch (node->type) {
        case AST_NUMBER:
            json_object_set_new(json_node, "Number", json_integer(node->data.ival));
            break;
        case AST_FLOAT:
            json_object_set_new(json_node, "Float", json_real(node->data.fval));
            break;
        case AST_STRING:
            json_object_set_new(json_node, "String", json_string(node->data.sval));
            break;
        case AST_IDENTIFIER: {
            if (node->data.sval) {
                json_object_set_new(json_node, "Identifier", json_string(node->data.sval));
                SymbolTableEntry *entry = find_symbol(symbol_table, node->data.sval);
                if (entry) {
                    json_object_set_new(json_node, "IdentifierType", json_string(var_type_to_string(entry->type)));
                } else {
                    json_object_set_new(json_node, "Identifier", json_string("Type Unknown"));
                }
            } else {
                json_object_set_new(json_node, "Identifier", json_string("null"));
            }
            break;
        }
        case AST_TYPE:
            json_object_set_new(json_node, "Type", json_string(node->data.sval));
            break;
        case AST_ADD:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        case AST_COMMA:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        case AST_SUB:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        case AST_CR:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            break;
        case AST_OB:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            break;
        case AST_PR:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            break;
        case AST_PVG:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            break;
        case AST_PWR:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        case AST_LIAB:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        case AST_DIS:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        case AST_IMM:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        case AST_AND:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        case AST_CONDITION:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            break;
        case AST_CLAUSE_EXPRESSION:
            if (node->data.clause.condition) {
                json_object_set_new(json_node, "Condition", ast_to_json(node->data.clause.condition, level + 1));
            } else {
                json_object_set_new(json_node, "Condition", json_string("NULL"));
            }

            if (node->data.clause.consequence) {
                json_object_set_new(json_node, "Consequence", ast_to_json(node->data.clause.consequence, level + 1));
            } else {
                json_object_set_new(json_node, "Consequence", json_string("NULL"));
            }
            break;
        case AST_ASSET_EXPRESSION:
            json_object_set_new(json_node, "Type", ast_to_json(node->data.asset.type, level + 1));
            json_object_set_new(json_node, "SubType", ast_to_json(node->data.asset.subtype, level + 1));
            json_object_set_new(json_node, "Subject1", ast_to_json(node->data.asset.subject1, level + 1));
            json_object_set_new(json_node, "Description", ast_to_json(node->data.asset.description, level + 1));
            json_object_set_new(json_node, "Subject2", ast_to_json(node->data.asset.subject2, level + 1));
            break;
        case AST_SUBJECT_EXPRESSION:
            json_object_set_new(json_node, "Description1", ast_to_json(node->data.subject.description1, level + 1));
            json_object_set_new(json_node, "Description2", ast_to_json(node->data.subject.description2, level + 1));
            json_object_set_new(json_node, "Location", ast_to_json(node->data.subject.location, level + 1));
            break;
        case AST_QUERY:
            json_object_set_new(json_node, "Query", json_string("Not Implemented"));
            break;
        case AST_DECLARATIONS:
            json_object_set_new(json_node, "Declarations", ast_to_json(node->data.declaration.type, level + 1));
            json_object_set_new(json_node, "Identifier", ast_to_json(node->data.declaration.identifier, level + 1));
            json_object_set_new(json_node, "Expression", ast_to_json(node->data.declaration.expression, level + 1));
            break;
        case AST_DECLARATION:
            if (node->data.declaration.type) {
                json_object_set_new(json_node, "Type", ast_to_json(node->data.declaration.type, level + 1));
            }
            if (node->data.declaration.identifier) {
                json_object_set_new(json_node, "Identifier", ast_to_json(node->data.declaration.identifier, level + 1));
            }
            if (node->data.declaration.expression) {
                json_object_set_new(json_node, "Expression", ast_to_json(node->data.declaration.expression, level + 1));
            }
            break;
        case AST_BINARY_OP:
            json_object_set_new(json_node, "Left", ast_to_json(node->data.binary_op.left, level + 1));
            json_object_set_new(json_node, "Right", ast_to_json(node->data.binary_op.right, level + 1));
            break;
        default:
            json_object_set_new(json_node, "Info", json_string("Unhandled node type"));
            break;
    }

    if (node->next != NULL) {
        json_object_set_new(json_node, "Next", ast_to_json(node->next, level));
    }

    return json_node;
}

const char* node_type_to_string(ASTNodeType type) {
    switch (type) {
        case AST_NUMBER: return "AST_NUMBER";
        case AST_FLOAT: return "AST_FLOAT";
        case AST_STRING: return "AST_STRING";
        case AST_IDENTIFIER: return "AST_IDENTIFIER";
        case AST_TYPE: return "AST_TYPE";
        case AST_ADD: return "AST_ADD";
        case AST_COMMA: return "AST_COMMA";
        case AST_SUB: return "AST_SUB";
        case AST_CR: return "AST_CR";
        case AST_OB: return "AST_OB";
        case AST_PR: return "AST_PR";
        case AST_PVG: return "AST_PVG";
        case AST_PWR: return "AST_PWR";
        case AST_LIAB: return "AST_LIAB";
        case AST_DIS: return "AST_DIS";
        case AST_IMM: return "AST_IMM";
        case AST_AND: return "AST_AND";
        case AST_CONDITION: return "AST_CONDITION";
        case AST_CLAUSE_EXPRESSION: return "AST_CLAUSE_EXPRESSION";
        case AST_ASSET_EXPRESSION: return "AST_ASSET_EXPRESSION";
        case AST_SUBJECT_EXPRESSION: return "AST_SUBJECT_EXPRESSION";
        case AST_QUERY: return "AST_QUERY";
        case AST_DECLARATIONS: return "AST_DECLARATIONS";
        case AST_DECLARATION: return "AST_DECLARATION";
        case AST_BINARY_OP: return "AST_BINARY_OP";
        default: return "Unknown ASTNodeType";
    }
}
