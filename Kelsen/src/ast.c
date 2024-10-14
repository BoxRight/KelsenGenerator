#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "symbol_table.h"

extern SymbolTable *symbol_table;

ASTNode *create_number(int value) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_NUMBER;
    node->var_type = TYPE_INT;
    node->data.ival = value;
    node->data.strval = malloc(20);  // Allocate memory for the string representation
    sprintf(node->data.strval, "%d", value);
    node->next = NULL;
    return node;
}

ASTNode *create_string(char *value) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_STRING;
    node->var_type = TYPE_STRING;
    node->data.sval = strdup(value);
    node->data.strval = strdup(value);
    node->next = NULL;
    return node;
}

ASTNode *create_identifier(char *name) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_IDENTIFIER;
    node->var_type = TYPE_UNKNOWN;
    node->data.sval = strdup(name);
    node->data.strval = strdup(name);
    node->next = NULL;
    return node;
}

ASTNode *create_float(double value) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_FLOAT;
    node->var_type = TYPE_FLOAT;
    node->data.fval = value;
    node->next = NULL;
    return node;
}

ASTNode *create_type_node(const char *type_name) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_TYPE;
    node->data.sval = strdup(type_name);
    node->next = NULL;  // Initialize next to NULL
    
    if (strcmp(type_name, "int") == 0) {
        node->var_type = TYPE_INT;
    } else if (strcmp(type_name, "float") == 0) {
        node->var_type = TYPE_FLOAT;
    } else if (strcmp(type_name, "string") == 0) {
        node->var_type = TYPE_STRING;
    } else if (strcmp(type_name, "asset") == 0) {
        node->var_type = TYPE_ASSET;
    } else if (strcmp(type_name, "subject") == 0) {
        node->var_type = TYPE_SUBJECT;
    } else if (strcmp(type_name, "clause") == 0) {
        node->var_type = TYPE_CLAUSE;
    } else if (strcmp(type_name, "query") == 0) {
        node->var_type = TYPE_QUERY;
    } else {
        node->var_type = TYPE_UNKNOWN;
    }

    return node;
}


ASTNode *create_binary_op(ASTNodeType type, ASTNode *left, ASTNode *right) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = type;
    node->var_type = TYPE_UNKNOWN;
    node->data.binary_op.left = left;
    node->data.binary_op.right = right;
    node->next = NULL;
    return node;
}

ASTNode *create_clause_expression(ASTNode *condition, ASTNode *consequence) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_CLAUSE_EXPRESSION;
    node->var_type = TYPE_UNKNOWN;
    node->data.clause.condition = condition;
    node->data.clause.consequence = consequence;
    node->next = NULL;
    return node;
}

ASTNode *create_legal(ASTNodeType type, ASTNode *expr) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = type;
    node->data.legal.expression = expr;
    switch (type) {
        case AST_CR:
            node->var_type = TYPE_CLAIM_RIGHT;
            break;
        case AST_OB:
            node->var_type = TYPE_OBLIGATION;
            break;
        case AST_PR:
            node->var_type = TYPE_PROHIBITION;
            break;
        case AST_PVG:
            node->var_type = TYPE_PRIVILEGE;
            break;
        case AST_PWR:
            node->var_type = TYPE_POWER;
            break;
        case AST_LIAB:
            node->var_type = TYPE_LIABILITY;
            break;
        case AST_DIS:
            node->var_type = TYPE_DISABILITY;
            break;
        case AST_IMM:
            node->var_type = TYPE_IMMUNITY;
            break;
        default:
            node->var_type = TYPE_UNKNOWN;
            break;
    }
    node->next = NULL;
    return node;
}




ASTNode *create_condition(ASTNode *expr) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_CONDITION;
    node->data.legal.expression = expr;
    node->next = NULL;
    return node;
}

ASTNode *create_consequence(ASTNode *expr) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_CONDITION; // Use the correct type if different
    node->data.legal.expression = expr;
    node->next = NULL;
    return node;
}

ASTNode *create_query(ASTNode *expression) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_QUERY;
    node->var_type = TYPE_QUERY;
    node->data.query.expression = expression;
    node->next = NULL;
    return node;
}


ASTNode *create_asset_expression(ASTNode *type, ASTNode *subtype, ASTNode *subject1, ASTNode *description, ASTNode *subject2) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_ASSET_EXPRESSION;
    node->var_type = TYPE_ASSET;
    node->data.asset.type = type;
    node->data.asset.subtype = subtype;
    node->data.asset.subject1 = subject1;
    node->data.asset.description = description;
    node->data.asset.subject2 = subject2;
    node->next = NULL;
    return node;
}

ASTNode *create_subject_expression(ASTNode *description1, ASTNode *description2, int age, ASTNode *location) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_SUBJECT_EXPRESSION;
    node->var_type = TYPE_SUBJECT;
    node->data.subject.description1 = description1;
    node->data.subject.description2 = description2;
    node->data.subject.age = age;
    node->data.subject.location = location;
    node->next = NULL;
    return node;
}

ASTNode *create_declarations(ASTNode *declarations, ASTNode *declaration) {
    if (!declarations) return declaration;

    ASTNode *node = declarations;
    while (node->next) {
        node = node->next;
    }
    node->next = declaration;
    return declarations;
}

ASTNode *create_declaration(ASTNode *type, ASTNode *identifier, ASTNode *expression) {
    ASTNode *node = (ASTNode *)malloc(sizeof(ASTNode));
    node->type = AST_DECLARATION;
    node->var_type = TYPE_UNKNOWN; // Set appropriate type if known
    node->data.declaration.type = type;
    node->data.declaration.identifier = identifier;
    node->data.declaration.expression = expression;
    node->next = NULL;
    return node;
}

void print_ast(ASTNode *node, int level) {
    if (!node) return;

    for (int i = 0; i < level; i++) {
        printf("  "); // Indentation
    }

    switch (node->type) {
        case AST_NUMBER:
            printf("Number: %d\n", node->data.ival);
            break;
        case AST_FLOAT:
            printf("Float: %f\n", node->data.fval);
            break;
        case AST_STRING:
            printf("String: %s\n", node->data.sval);
            break;
        case AST_IDENTIFIER: {
            if (node->data.sval) {
                printf("Identifier: %s\n", node->data.sval);
                SymbolTableEntry *entry = find_symbol(symbol_table, node->data.sval);
                if (entry) {
                    printf("Identifier: %s (Type: %s)\n", node->data.sval, var_type_to_string(entry->type));
                } else {
                    printf("Identifier: %s (Type: Unknown)\n", node->data.sval);
                }
            } else {
                printf("Identifier: (null)\n");
            }
            break;
        }
        case AST_TYPE:
            printf("Type: %s\n", node->data.sval);
            break;
        case AST_ADD:
            printf("Add:\n");
            print_ast(node->data.binary_op.left, level + 1);
            print_ast(node->data.binary_op.right, level + 1);
            break;
        case AST_COMMA:
			printf("Comma:\n");
			print_ast(node->data.binary_op.left, level + 1);
			print_ast(node->data.binary_op.right, level + 1);
			break;
        case AST_SUB:
            printf("Sub:\n");
            print_ast(node->data.binary_op.left, level + 1);
            print_ast(node->data.binary_op.right, level + 1);
            break;
        case AST_CR:
            printf("Claim Right:\n");
            print_ast(node->data.binary_op.left, level + 1);
            break;
        case AST_OB:
            printf("Obligation:\n");
            print_ast(node->data.binary_op.left, level + 1);
            break;
        case AST_PR:
            printf("Prohibition:\n");
            print_ast(node->data.binary_op.left, level + 1);
            break;
        case AST_PVG:
            printf("Privilege:\n");
            print_ast(node->data.binary_op.left, level + 1);
            break;
        case AST_PWR:
            printf("Power:\n");
            print_ast(node->data.binary_op.left, level + 1);
            print_ast(node->data.binary_op.right, level + 1);
            break;
        case AST_LIAB:
            printf("Liability:\n");
            print_ast(node->data.binary_op.left, level + 1);
            print_ast(node->data.binary_op.right, level + 1);
            break;
        case AST_DIS:
            printf("Disability:\n");
            print_ast(node->data.binary_op.left, level + 1);
            print_ast(node->data.binary_op.right, level + 1);
            break;
        case AST_IMM:
            printf("Immunity:\n");
            print_ast(node->data.binary_op.left, level + 1);
            print_ast(node->data.binary_op.right, level + 1);
            break;
        case AST_AND:
            printf("And:\n");
            print_ast(node->data.binary_op.left, level + 1);
            print_ast(node->data.binary_op.right, level + 1);
            break;
        case AST_CONDITION:
            printf("Condition:\n");
            print_ast(node->data.binary_op.left, level + 1);
            break;
        case AST_CLAUSE_EXPRESSION:
            printf("Clause Expression:\n");
            if (node->data.clause.condition) {
                printf("Condition:\n");
                if (node->data.clause.condition->type == AST_AND) {
                    printf("Condition is AND\n");
                    print_ast(node->data.clause.condition->data.binary_op.left, level + 1);
                    print_ast(node->data.clause.condition->data.binary_op.right, level + 1);
                } else {
                    printf("Condition type: %d\n", node->data.clause.condition->type);
                    print_ast(node->data.clause.condition, level + 1);
                }
            } else {
                printf("Condition is NULL\n");
            }

            if (node->data.clause.consequence) {
                printf("Consequence:\n");
                print_ast(node->data.clause.consequence, level + 1);
            } else {
                printf("Consequence is NULL\n");
            }
            break;
        case AST_ASSET_EXPRESSION:
            printf("Asset Expression:\n");
            print_ast(node->data.asset.type, level + 1);
            printf("Type next: %p\n", (void *)node->data.asset.type->next);
            print_ast(node->data.asset.subtype, level + 1);
            printf("Subtype next: %p\n", (void *)node->data.asset.subtype->next);
            print_ast(node->data.asset.subject1, level + 1);
            printf("Subject1 next: %p\n", (void *)node->data.asset.subject1->next);
            print_ast(node->data.asset.description, level + 1);
            printf("Description next: %p\n", (void *)node->data.asset.description->next);
            print_ast(node->data.asset.subject2, level + 1);
            printf("Subject2 next: %p\n", (void *)node->data.asset.subject2->next);
            break;
        case AST_SUBJECT_EXPRESSION:
            printf("Subject Expression:\n");
            print_ast(node->data.subject.description1, level + 1);
            print_ast(node->data.subject.description2, level + 1);
            print_ast(node->data.subject.location, level + 1);
            break;
        case AST_QUERY:
            printf("Query:\n");
            break;
        case AST_DECLARATIONS:
            printf("Declarations:\n");
            if (node->data.declaration.type) print_ast(node->data.declaration.type, level + 1);
            if (node->data.declaration.identifier) print_ast(node->data.declaration.identifier, level + 1);
            if (node->data.declaration.expression) print_ast(node->data.declaration.expression, level + 1);
            break;
        case AST_DECLARATION:
            printf("Declaration:\n");
            if (node->data.declaration.type) {
                printf("  Type: %s\n", node->data.declaration.type->data.sval);
            }
            if (node->data.declaration.identifier) {
                printf("  Identifier: %s\n", node->data.declaration.identifier->data.sval);
            }
            if (node->data.declaration.expression) {
                print_ast(node->data.declaration.expression, level + 1);
            }
            break;
        case AST_BINARY_OP:
            printf("Binary Operation:\n");
            print_ast(node->data.binary_op.left, level + 1);
            print_ast(node->data.binary_op.right, level + 1);
            break;
        default: {
            if (node->data.sval) {
                SymbolTableEntry *entry = find_symbol(symbol_table, node->data.sval);
                if (entry) {
                    printf("Identifier: %s (Type: %s)\n", node->data.sval, var_type_to_string(entry->type));
                } else {
                    printf("Unknown AST Node Type: %d\n", node->type);
                }
            } else {
                printf("Unknown AST Node Type: %d\n", node->type);
            }
            break;
        }
    }

    if (node->next != NULL) {
        printf("Next node exists: %p\n", (void *)node->next);
        print_ast(node->next, level);
    } else {
        printf("Next node is NULL\n");
    }
}

void free_ast(ASTNode *node) {
    if (node == NULL) return;

    switch (node->type) {
        case AST_STRING:
            free(node->data.sval);
            break;

        case AST_SUBJECT_EXPRESSION:
            free(node->data.subject.description1->data.sval);
            free(node->data.subject.description2->data.sval);
            free(node->data.subject.location->data.sval);
            free_ast(node->data.subject.description1);
            free_ast(node->data.subject.description2);
            free_ast(node->data.subject.location);
            break;

        case AST_ASSET_EXPRESSION:
            free(node->data.asset.type->data.sval);
            free(node->data.asset.subtype->data.sval);
            free(node->data.asset.subject1->data.sval);
            free(node->data.asset.description->data.sval);
            free(node->data.asset.subject2->data.sval);
            free_ast(node->data.asset.type);
            free_ast(node->data.asset.subtype);
            free_ast(node->data.asset.subject1);
            free_ast(node->data.asset.description);
            free_ast(node->data.asset.subject2);
            break;

        case AST_CLAUSE_EXPRESSION:
            free(node->data.clause.condition->data.binary_op.left->data.sval);
            free(node->data.clause.condition->data.binary_op.right->data.sval);
            free_ast(node->data.clause.condition->data.binary_op.left);
            free_ast(node->data.clause.condition->data.binary_op.right);
            free_ast(node->data.clause.consequence);
            break;

        case AST_QUERY:
            break;


    }

    // Free the next node
    if (node->next) {
        free_ast(node->next);
    }

    // Free the current node
    free(node);
}

