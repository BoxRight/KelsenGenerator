#ifndef AST_H
#define AST_H

#include "types.h"

typedef enum {
    AST_NUMBER,
    AST_FLOAT,
    AST_STRING,
    AST_IDENTIFIER,
    AST_TYPE,
    AST_ADD,
    AST_SUB,
    AST_CR,
    AST_OB,
    AST_PR,
    AST_PVG,
    AST_PWR,
    AST_LIAB,
    AST_DIS,
    AST_IMM,
    AST_AND,
    AST_CONDITION,  // Add this line
    AST_COMMA, // Added AST_COMMA
    AST_DECLARATIONS, // Added for declarations
    AST_DECLARATION, // Added for single declaration
    AST_ASSET_EXPRESSION,
    AST_SUBJECT_EXPRESSION,
    AST_CLAUSE_EXPRESSION,
    AST_QUERY,
    AST_CLAUSE,
    AST_LEGAL,
    AST_BINARY_OP // Added AST_BINARY_OP
} ASTNodeType;

typedef struct ASTNode {
    ASTNodeType type;
    VarType var_type;
    union {
        int ival;
        double fval;
        char *sval;
        char *strval;  
        struct {
            struct ASTNode *left;
            struct ASTNode *right;
        } binary_op;
        struct {
            struct ASTNode *type;
            struct ASTNode *subtype;
            struct ASTNode *subject1;
            struct ASTNode *description;
            struct ASTNode *subject2;
        } asset;
        struct {
            struct ASTNode *description1;
            struct ASTNode *description2;
            int age;
            struct ASTNode *location;
        } subject;
        struct {
            struct ASTNode *expression;
        } query;
        struct {
            struct ASTNode *condition;
            struct ASTNode *consequence;
        } clause;
        struct {
            struct ASTNode *expression;
        } legal;
        struct {
            struct ASTNode *type;
            struct ASTNode *identifier;
            struct ASTNode *expression;
        } declaration;
    } data;
    struct ASTNode *next; // For linking nodes in lists, if necessary
} ASTNode;

// Function declarations
ASTNode *create_number(int value);
ASTNode *create_float(double value);
ASTNode *create_string(char *value);
ASTNode *create_identifier(char *name);
ASTNode *create_type_node(const char *type_name);
ASTNode *create_binary_op(ASTNodeType type, ASTNode *left, ASTNode *right);
ASTNode *create_clause_expression(ASTNode *condition, ASTNode *consequence);
ASTNode *create_legal(ASTNodeType type, ASTNode *expr);
ASTNode *create_condition(ASTNode *expr);
ASTNode *create_consequence(ASTNode *expr);
ASTNode *create_asset_expression(ASTNode *type, ASTNode *subtype, ASTNode *subject1, ASTNode *description, ASTNode *subject2);
ASTNode *create_subject_expression(ASTNode *description1, ASTNode *description2, int age, ASTNode *location);
ASTNode *create_query(ASTNode *expression); 
ASTNode *create_declarations(ASTNode *declarations, ASTNode *declaration);
ASTNode *create_declaration(ASTNode *type, ASTNode *identifier, ASTNode *expression);
void free_ast(ASTNode *node);

const char* var_type_to_string(VarType type);
void print_ast(ASTNode *node, int level); // Added print_ast function declaration
const char* node_type_to_string(ASTNodeType type);


#endif // AST_H

