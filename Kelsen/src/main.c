#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "parser.tab.h"  // Generated from Bison
#include "symbol_table.h"
#include "ast_to_json.c"
#include "contract_generator.h"

extern SymbolTable *symbol_table;
extern ASTNode *root;
extern FILE *yyin;


int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    yyin = fopen(argv[1], "r");
    if (!yyin) {
        perror(argv[1]);
        return EXIT_FAILURE;
    }

    symbol_table = create_symbol_table();
    if (yyparse() != 0) {
        fprintf(stderr, "Failed to parse.\n");
        fclose(yyin);
        free_symbol_table(symbol_table);
        return EXIT_FAILURE;
    }
    fclose(yyin);

    json_t *root_json = ast_to_json(root, 0);
    if (root_json == NULL) {
        fprintf(stderr, "Failed to convert AST to JSON.\n");
        free_symbol_table(symbol_table);
        free_ast(root);
        return EXIT_FAILURE;
    }

    // Write the JSON to a file in the output directory
    FILE *json_file = fopen("../output/ast_output.json", "w");
    if (json_file == NULL) {
        perror("Failed to open output file");
        json_decref(root_json);
        free_symbol_table(symbol_table);
        free_ast(root);
        return EXIT_FAILURE;
    }

    char *json_output = json_dumps(root_json, JSON_INDENT(2));
    if (json_output != NULL) {
        fprintf(json_file, "%s\n", json_output);
        free(json_output);
    } else {
        fprintf(stderr, "Failed to serialize JSON.\n");
    }

    fclose(json_file);


    // Clean up
    json_decref(root_json);
    free_symbol_table(symbol_table);
    free_ast(root);  // Ensure this function properly frees all AST nodes

    return EXIT_SUCCESS;
}
