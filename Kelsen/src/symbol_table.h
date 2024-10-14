#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

#include "uthash.h"
#include "ast.h"

typedef struct SymbolTableEntry {
    char *name;
    VarType type;
    UT_hash_handle hh; // makes this structure hashable
} SymbolTableEntry;

typedef struct {
    SymbolTableEntry *symbols; // a hash table of symbols
} SymbolTable;

extern SymbolTable *create_symbol_table();
void free_symbol_table(SymbolTable *table);
int insert_symbol(SymbolTable *table, const char *name, VarType type);
SymbolTableEntry *find_symbol(SymbolTable *table, const char *name);
void print_symbol_table(SymbolTable *table);

const char* var_type_to_string(VarType type); // Add this line

int check_declaration_types(ASTNode *declaration);
int check_expression_types(ASTNode *expression, VarType expected_type);
VarType get_expression_type(struct ASTNode *expression);

#endif // SYMBOL_TABLE_H
