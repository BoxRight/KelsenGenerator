#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symbol_table.h"
#include "uthash.h"

extern SymbolTable *symbol_table;
// Create a new symbol table
SymbolTable *create_symbol_table() {
    SymbolTable *table = (SymbolTable *)malloc(sizeof(SymbolTable));
    table->symbols = NULL; // Initialize the hash table
    return table;
}

int insert_symbol(SymbolTable *table, const char *name, VarType type) {
    SymbolTableEntry *entry = find_symbol(table, name);
    if (entry) {
        printf("Symbol %s already exists in the symbol table with type %d (%s).\n", name, entry->type, var_type_to_string(entry->type));
        return 0; // Symbol already exists
    }
    entry = (SymbolTableEntry *)malloc(sizeof(SymbolTableEntry));
    entry->name = strdup(name);
    entry->type = type;
    HASH_ADD_KEYPTR(hh, table->symbols, entry->name, strlen(entry->name), entry);
    return 1; // Symbol inserted successfully
}



// Find a symbol in the symbol table
SymbolTableEntry* find_symbol(SymbolTable *table, const char *name) {
    SymbolTableEntry *entry = NULL;
    HASH_FIND_STR(table->symbols, name, entry);
    return entry;
}

// Free the symbol table
void free_symbol_table(SymbolTable *table) {
    SymbolTableEntry *entry, *tmp;
    HASH_ITER(hh, table->symbols, entry, tmp) {
        HASH_DEL(table->symbols, entry);
        free(entry->name);
        free(entry);
    }
    free(table);
}

// Print the contents of the symbol table
void print_symbol_table(SymbolTable *table) {
    SymbolTableEntry *entry, *tmp;
    printf("Symbol Table:\n");
    HASH_ITER(hh, table->symbols, entry, tmp) {
        printf("Name: %s, Type: %d\n", entry->name, entry->type);
    }
}

int check_declaration_types(ASTNode *declaration) {
    if (!declaration) return 0;
    ASTNode *identifier_node = declaration->data.declaration.identifier;
    ASTNode *type_node = declaration->data.declaration.type;
    ASTNode *expression_node = declaration->data.declaration.expression;

    if (expression_node) {
        VarType expr_type = get_expression_type(expression_node);
        // If the expected type is TYPE_QUERY, check for any query-related types
        if (type_node->var_type == TYPE_QUERY) {
            if (expr_type != TYPE_POWER && expr_type != TYPE_LIABILITY && expr_type != TYPE_DISABILITY && expr_type != TYPE_IMMUNITY) {
                fprintf(stderr, "Type error in declaration: expected TYPE_QUERY but got %d (%s)\n", expr_type, var_type_to_string(expr_type));
                return 0;
            }
        } else if (expr_type != type_node->var_type) {
            fprintf(stderr, "Type error in declaration: expected type %d (%s) but got type %d (%s)\n",
                type_node->var_type, var_type_to_string(type_node->var_type),
                expr_type, var_type_to_string(expr_type));
            return 0;
        }
    }

    return 1;
}


int check_expression_types(ASTNode *expression, VarType expected_type) {
    if (!expression) {

        return 0;
    }

    VarType expr_type = get_expression_type(expression);

        expression->data.sval ? expression->data.sval : "(null)",
        expr_type, var_type_to_string(expr_type),
        expected_type, var_type_to_string(expected_type);

    // Type check for TYPE_CLAUSE
    if (expected_type == TYPE_CLAUSE && expr_type != TYPE_CLAUSE) {

            expected_type, var_type_to_string(expected_type),
            expr_type, var_type_to_string(expr_type),
            expression->data.sval ? expression->data.sval : "(null)";
        return 0;
    }

    // Type check for TYPE_LEGAL
    if (expected_type == TYPE_LEGAL) {
        if (expr_type != TYPE_CLAIM_RIGHT && expr_type != TYPE_OBLIGATION && expr_type != TYPE_PROHIBITION &&
            expr_type != TYPE_PRIVILEGE && expr_type != TYPE_POWER && expr_type != TYPE_LIABILITY &&
            expr_type != TYPE_DISABILITY && expr_type != TYPE_IMMUNITY) {

                expected_type, var_type_to_string(expected_type),
                expr_type, var_type_to_string(expr_type),
                expression->data.sval ? expression->data.sval : "(null)";
            return 0;
        }
    }

    // Type check for TYPE_QUERY
    if (expected_type == TYPE_QUERY) {
        if (expr_type != TYPE_POWER && expr_type != TYPE_LIABILITY && expr_type != TYPE_DISABILITY && expr_type != TYPE_IMMUNITY) {

                    expected_type, var_type_to_string(expected_type), 
                    expr_type, var_type_to_string(expr_type),
                    expression->data.sval ? expression->data.sval : "(null)";
            return 0;
        }
    }

    // General type check for other types
    if (expected_type != TYPE_CLAUSE && expected_type != TYPE_LEGAL && expected_type != TYPE_QUERY && expr_type != expected_type) {

            expected_type, var_type_to_string(expected_type),
            expr_type, var_type_to_string(expr_type),
            expression->data.sval ? expression->data.sval : "(null)";
        return 0;
    }

    return 1;
}



const char* var_type_to_string(VarType type) {
    switch (type) {
        case TYPE_INT: return "TYPE_INT";
        case TYPE_FLOAT: return "TYPE_FLOAT";
        case TYPE_STRING: return "TYPE_STRING";
        case TYPE_ASSET: return "TYPE_ASSET";
        case TYPE_SUBJECT: return "TYPE_SUBJECT";
        case TYPE_CLAUSE: return "TYPE_CLAUSE";
        case TYPE_QUERY: return "TYPE_QUERY";
        case TYPE_CONDITION: return "TYPE_CONDITION";
        case TYPE_CLAIM_RIGHT: return "TYPE_CLAIM_RIGHT";
        case TYPE_OBLIGATION: return "TYPE_OBLIGATION";
        case TYPE_PROHIBITION: return "TYPE_PROHIBITION";
        case TYPE_PRIVILEGE: return "TYPE_PRIVILEGE";
        case TYPE_POWER: return "TYPE_POWER";
        case TYPE_LIABILITY: return "TYPE_LIABILITY";
        case TYPE_DISABILITY: return "TYPE_DISABILITY";
        case TYPE_IMMUNITY: return "TYPE_IMMUNITY";
        case TYPE_LEGAL: return "TYPE_LEGAL";
        case TYPE_UNKNOWN: return "TYPE_UNKNOWN";
        default: return "UNKNOWN_TYPE";
    }
}


VarType get_expression_type(ASTNode *expression) {
    if (!expression) {
        return TYPE_UNKNOWN;
    }

    VarType type = TYPE_UNKNOWN;
    switch (expression->type) {
        case AST_NUMBER:
            type = TYPE_INT;
            break;
        case AST_FLOAT:
            type = TYPE_FLOAT;
            break;
        case AST_STRING:
            type = TYPE_STRING;
            break;
        case AST_IDENTIFIER: {
            SymbolTableEntry *entry = find_symbol(symbol_table, expression->data.sval);
            return entry ? entry->type : TYPE_UNKNOWN;
        }
        case AST_BINARY_OP: {
            VarType left_type = get_expression_type(expression->data.binary_op.left);
            VarType right_type = get_expression_type(expression->data.binary_op.right);
            if (left_type == right_type) {
                type = left_type;
            }
            break;
        }
        case AST_ASSET_EXPRESSION:
            type = TYPE_ASSET;
            break;
        case AST_SUBJECT_EXPRESSION:
            type = TYPE_SUBJECT;
            break;
        case AST_CLAUSE_EXPRESSION:
            type = TYPE_CLAUSE;
            break;
        case AST_CR:
            type = TYPE_CLAIM_RIGHT;
            break;
        case AST_OB:
            type = TYPE_OBLIGATION;
            break;
        case AST_PR:
            type = TYPE_PROHIBITION;
            break;
        case AST_PVG:
            type = TYPE_PRIVILEGE;
            break;
        case AST_PWR:
            type = TYPE_POWER;
            break;
        case AST_LIAB:
            type = TYPE_LIABILITY;
            break;
        case AST_DIS:
            type = TYPE_DISABILITY;
            break;
        case AST_IMM:
            type = TYPE_IMMUNITY;
            break;
        case AST_LEGAL: {
            switch (expression->data.legal.expression->type) {
                case AST_CR:
                    type = TYPE_CLAIM_RIGHT;
                    break;
                case AST_OB:
                    type = TYPE_OBLIGATION;
                    break;
                case AST_PR:
                    type = TYPE_PROHIBITION;
                    break;
                case AST_PVG:
                    type = TYPE_PRIVILEGE;
                    break;
                case AST_PWR:
                    type = TYPE_POWER;
                    break;
                case AST_LIAB:
                    type = TYPE_LIABILITY;
                    break;
                case AST_DIS:
                    type = TYPE_DISABILITY;
                    break;
                case AST_IMM:
                    type = TYPE_IMMUNITY;
                    break;
                default:
                    type = TYPE_UNKNOWN;
                    break;
            }
            break;
        }
        case AST_CONDITION:
            type = TYPE_CONDITION;
            break;
        case AST_QUERY:
            type = TYPE_QUERY;
            break;
        default:
            type = TYPE_UNKNOWN;
            break;
    }
    // Debug print

    return type;
}

