%{
#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "symbol_table.h"

ASTNode *root;  // Define the root variable
extern SymbolTable *symbol_table;

extern int yylex(void);
extern int yylineno;
extern char *yytext;
int check_declaration_types(ASTNode *declaration);
int check_expression_types(ASTNode *expression, VarType expected_type);



void yyerror(const char *s) {
    fprintf(stderr, "Error: %s at line %d, near token '%s'\n", s, yylineno, yytext);
}

#define YYDEBUG 1
%}


%union {
    int ival;
    double fval;
    char *sval;
    struct ASTNode *node;
}

%token <ival> TOKEN_INT_LITERAL
%token <fval> TOKEN_FLOAT_LITERAL
%token <sval> TOKEN_STRING_LITERAL
%token <sval> TOKEN_IDENTIFIER

%token TOKEN_TYPE_INT TOKEN_TYPE_FLOAT TOKEN_TYPE_STRING TOKEN_TYPE_ASSET TOKEN_TYPE_SUBJECT TOKEN_TYPE_CLAUSE TOKEN_TYPE_QUERY
%token TOKEN_ADD TOKEN_SUB TOKEN_PLUS TOKEN_MINUS TOKEN_ASSIGN
%token TOKEN_SEMICOLON TOKEN_COMMA
%token TOKEN_CR_OPEN TOKEN_OB_OPEN TOKEN_PR_OPEN TOKEN_PVG_OPEN
%token TOKEN_CLOSE_PAREN TOKEN_OPEN_BRACE TOKEN_CLOSE_BRACE
%token TOKEN_CONDITION TOKEN_CONSEQUENCE TOKEN_AND
%token TOKEN_PWR_OPEN TOKEN_LIAB_OPEN TOKEN_DIS_OPEN TOKEN_IMM_OPEN
%token TOKEN_UNKNOWN

%token TOKEN_SERVICE TOKEN_PROPERTY TOKEN_NM TOKEN_M

%type <node> program declarations declaration type identifier expression numeric_expression additive_expression string_expression asset_expression query_expression subject_expression clause_expression asset_type asset_subtype condition_expression
%type <node> claim_right obligation prohibition privilege power liability disability immunity

%left TOKEN_ADD TOKEN_SUB
%left TOKEN_AND

%%

program:
      declarations { root = $1; }
    ;

declarations:
      declaration { $$ = $1; }
    | declarations declaration { $$ = create_declarations($1, $2); }
    ;

declaration:
      type identifier TOKEN_ASSIGN expression TOKEN_SEMICOLON {
          $$ = create_declaration($1, $2, $4);
          if (!insert_symbol(symbol_table, $2->data.sval, $1->var_type)) {
              yyerror("Symbol insertion failed");
          }
          if (!check_declaration_types($$)) {
              yyerror("Type mismatch in declaration");
          }
      }
    | type identifier TOKEN_SEMICOLON {
          $$ = create_declaration($1, $2, NULL);
          if (!insert_symbol(symbol_table, $2->data.sval, $1->var_type)) {
              yyerror("Symbol insertion failed");
          }
      }
    ;

type:
      TOKEN_TYPE_INT { $$ = create_type_node("int"); }
    | TOKEN_TYPE_FLOAT { $$ = create_type_node("float"); }
    | TOKEN_TYPE_STRING { $$ = create_type_node("string"); }
    | TOKEN_TYPE_ASSET { $$ = create_type_node("asset"); }
    | TOKEN_TYPE_SUBJECT { $$ = create_type_node("subject"); }
    | TOKEN_TYPE_CLAUSE { $$ = create_type_node("clause"); }
    | TOKEN_TYPE_QUERY { $$ = create_type_node("query"); }
    ;



identifier:
      TOKEN_IDENTIFIER { $$ = create_identifier($1); }
    ;

expression:
      numeric_expression { $$ = $1; if (!check_expression_types($$, $1->var_type)) yyerror("Type mismatch in numeric expression"); }
    | additive_expression { $$ = $1; if (!check_expression_types($$, $1->var_type)) yyerror("Type mismatch in additive expression"); }
    | asset_expression { $$ = $1; if (!check_expression_types($$, TYPE_ASSET)) yyerror("Type mismatch in asset expression"); }
    | subject_expression { $$ = $1; if (!check_expression_types($$, TYPE_SUBJECT)) yyerror("Type mismatch in subject expression"); }
    | string_expression { $$ = $1; if (!check_expression_types($$, TYPE_STRING)) yyerror("Type mismatch in string expression"); }
    | clause_expression { $$ = $1; if (!check_expression_types($$, TYPE_CLAUSE)) yyerror("Type mismatch in clause expression"); }
    | query_expression { $$ = $1; if (!check_expression_types($$, TYPE_QUERY)) yyerror("Type mismatch in query expression"); };

numeric_expression:
      TOKEN_INT_LITERAL { $$ = create_number($1); }
    | TOKEN_FLOAT_LITERAL { $$ = create_float($1); }
    ;

additive_expression:
      numeric_expression TOKEN_ADD numeric_expression { $$ = create_binary_op(AST_ADD, $1, $3); if (!check_expression_types($$, TYPE_INT)) yyerror("Type error in additive expression"); }
    | numeric_expression TOKEN_SUB numeric_expression { $$ = create_binary_op(AST_SUB, $1, $3); if (!check_expression_types($$, TYPE_INT)) yyerror("Type error in additive expression"); }
    ;

string_expression:
      TOKEN_STRING_LITERAL { $$ = create_string($1); }
    ;


asset_expression:
      asset_type TOKEN_COMMA asset_subtype TOKEN_COMMA subject_expression TOKEN_COMMA string_expression TOKEN_COMMA subject_expression {
          // Check the types of each component
          if (!check_expression_types($5, TYPE_SUBJECT)) {
              yyerror("Type error: Expected subject type for the third part in asset expression");
          } else if (!check_expression_types($7, TYPE_STRING)) {
              yyerror("Type error: Expected string type for the fourth part in asset expression");
          } else if (!check_expression_types($9, TYPE_SUBJECT)) {
              yyerror("Type error: Expected subject type for the fifth part in asset expression");
          } else {
              // If all checks pass, create the AST node
        $$ = create_asset_expression($1, $3, $5, $7, $9);
            if (!check_expression_types($$, TYPE_ASSET)) {
                yyerror("Type error in asset expression");
              }
          }
      }
    | asset_type TOKEN_COMMA asset_subtype TOKEN_COMMA identifier TOKEN_COMMA identifier TOKEN_COMMA identifier {
          // Check the types of each component
          if (!check_expression_types($5, TYPE_SUBJECT)) {
              yyerror("Type error: Expected subject type for the third part in asset expression");
          } else if (!check_expression_types($7, TYPE_STRING)) {
              yyerror("Type error: Expected string type for the fourth part in asset expression");
          } else if (!check_expression_types($9, TYPE_SUBJECT)) {
              yyerror("Type error: Expected subject type for the fifth part in asset expression");
          } else {
              // If all checks pass, create the AST node
        $$ = create_asset_expression($1, $3, $5, $7, $9);
            if (!check_expression_types($$, TYPE_ASSET)) {
                yyerror("Type error in asset expression");
              }
          }
      }
    ;

subject_expression:
      string_expression TOKEN_COMMA string_expression TOKEN_COMMA numeric_expression TOKEN_COMMA string_expression {
          // Check the types of each component
          if (!check_expression_types($1, TYPE_STRING)) {
              yyerror("Type error: Expected string type for the first string expression in subject expression");
          } else if (!check_expression_types($3, TYPE_STRING)) {
              yyerror("Type error: Expected string type for the second string expression in subject expression");
          } else if (!check_expression_types($5, TYPE_INT)) {
              yyerror("Type error: Expected int type for the integer literal in subject expression");
          } else if (!check_expression_types($7, TYPE_STRING)) {
              yyerror("Type error: Expected string type for the third string expression in subject expression");
          } else {
              // If all checks pass, create the AST node
              $$ = create_subject_expression($1, $3, $5->data.ival, $7);
              if (!check_expression_types($$, TYPE_SUBJECT)) {
                  yyerror("Type error in subject expression");
              }
          }
      };



claim_right:
      TOKEN_CR_OPEN asset_expression TOKEN_CLOSE_PAREN { 

          if (!check_expression_types($2, TYPE_ASSET)) {
              yyerror("Type error in claim-right expression: expected asset");
          } else {
              $$ = create_legal(AST_CR, $2); 

          }
      }
    | TOKEN_CR_OPEN identifier TOKEN_CLOSE_PAREN { 

          if (!check_expression_types($2, TYPE_ASSET)) {
              yyerror("Type error in claim-right expression: expected asset");
          } else {
              $$ = create_legal(AST_CR, $2);

          }
      }
    ;

obligation:
      TOKEN_OB_OPEN asset_expression TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET)) {
              yyerror("Type error in obligation expression: expected asset");
          } else {
              $$ = create_legal(AST_OB, $2); 
              $$->var_type = TYPE_LEGAL;
          }
      }
    | TOKEN_OB_OPEN identifier TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET)) {
              yyerror("Type error in obligation expression: expected asset");
          } else {
              $$ = create_legal(AST_OB, $2); 
              $$->var_type = TYPE_LEGAL;
          }
      }
    ;

prohibition:
      TOKEN_PR_OPEN asset_expression TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET)) {
              yyerror("Type error in prohibition expression: expected asset");
          } else {
              $$ = create_legal(AST_PR, $2); 
              $$->var_type = TYPE_LEGAL;
          }
      }
    | TOKEN_PR_OPEN identifier TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET)) {
              yyerror("Type error in prohibition expression: expected asset");
          } else {
              $$ = create_legal(AST_PR, $2); 
              $$->var_type = TYPE_LEGAL;
          }
      }
    ;

privilege:
      TOKEN_PVG_OPEN asset_expression TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET)) {
              yyerror("Type error in privilege expression: expected asset");
          } else {
              $$ = create_legal(AST_PVG, $2); 
              $$->var_type = TYPE_LEGAL;
          }
      }
    | TOKEN_PVG_OPEN identifier TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET)) {
              yyerror("Type error in privilege expression: expected asset");
          } else {
              $$ = create_legal(AST_PVG, $2); 
              $$->var_type = TYPE_LEGAL;
          }
      }
    ;
condition_expression:
      asset_expression { 
          if (!check_expression_types($1, TYPE_ASSET)) {
              yyerror("Type error in condition expression: expected asset");
          } else {
              $$ = create_condition($1); 
              $$->var_type = TYPE_CONDITION;
          }
      }
    | identifier { 
          if (!check_expression_types($1, TYPE_ASSET)) {
              yyerror("Type error in condition expression: expected asset");
          } else {
              $$ = create_condition($1); 
              $$->var_type = TYPE_CONDITION;
          }
      }
    | asset_expression TOKEN_AND asset_expression { 
          if (!check_expression_types($1, TYPE_ASSET) || !check_expression_types($3, TYPE_ASSET)) {
              yyerror("Type error in condition expression: expected asset");
          } else {
              $$ = create_condition(create_binary_op(AST_AND, $1, $3)); 
              $$->var_type = TYPE_CONDITION;
          }
      }
    | identifier TOKEN_AND identifier { 
          if (!check_expression_types($1, TYPE_ASSET) || !check_expression_types($3, TYPE_ASSET)) {
              yyerror("Type error in condition expression: expected asset");
          } else {
              $$ = create_condition(create_binary_op(AST_AND, $1, $3)); 
              $$->var_type = TYPE_CONDITION;
          }
      }
    ;




clause_expression:
    TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA claim_right TOKEN_CLOSE_BRACE { 
        if (!check_expression_types($2, TYPE_CONDITION) || !check_expression_types($4, TYPE_CLAIM_RIGHT)) {
            yyerror("Type error in clause expression");
            YYERROR;
        } else {
            $$ = create_clause_expression($2, $4); 
            $$->var_type = TYPE_CLAUSE;
        }
    }
    | TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA obligation TOKEN_CLOSE_BRACE { 
        if (!check_expression_types($2, TYPE_CONDITION) || !check_expression_types($4, TYPE_OBLIGATION)) {
            yyerror("Type error in clause expression");
        } else {
            $$ = create_clause_expression($2, $4); 
            $$->var_type = TYPE_CLAUSE;
        }
    }
    | TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA prohibition TOKEN_CLOSE_BRACE { 
        if (!check_expression_types($2, TYPE_CONDITION) || !check_expression_types($4, TYPE_PROHIBITION)) {
            yyerror("Type error in clause expression");
        } else {
            $$ = create_clause_expression($2, $4); 
            $$->var_type = TYPE_CLAUSE;
        }
    }
    | TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA privilege TOKEN_CLOSE_BRACE { 
        if (!check_expression_types($2, TYPE_CONDITION) || !check_expression_types($4, TYPE_PRIVILEGE)) {
            yyerror("Type error in clause expression");
        } else {
            $$ = create_clause_expression($2, $4); 
            $$->var_type = TYPE_CLAUSE;
        }
    }
    | TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA power TOKEN_CLOSE_BRACE { 
        if (!check_expression_types($2, TYPE_CONDITION) || !check_expression_types($4, TYPE_POWER)) {
            yyerror("Type error in clause expression");
        } else {
            $$ = create_clause_expression($2, $4); 
            $$->var_type = TYPE_CLAUSE;
        }
    }
    | TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA liability TOKEN_CLOSE_BRACE { 
        if (!check_expression_types($2, TYPE_CONDITION) || !check_expression_types($4, TYPE_LIABILITY)) {
            yyerror("Type error in clause expression");
        } else {
            $$ = create_clause_expression($2, $4); 
            $$->var_type = TYPE_CLAUSE;
        }
    }
    | TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA disability TOKEN_CLOSE_BRACE { 
        if (!check_expression_types($2, TYPE_CONDITION) || !check_expression_types($4, TYPE_DISABILITY)) {
            yyerror("Type error in clause expression");
        } else {
            $$ = create_clause_expression($2, $4); 
            $$->var_type = TYPE_CLAUSE;
        }
    }
    | TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA immunity TOKEN_CLOSE_BRACE { 
        if (!check_expression_types($2, TYPE_CONDITION) || !check_expression_types($4, TYPE_IMMUNITY)) {
            yyerror("Type error in clause expression");
        } else {
            $$ = create_clause_expression($2, $4); 
            $$->var_type = TYPE_CLAUSE;
        }
    }
    ;


power:
      TOKEN_PWR_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA claim_right TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_CLAIM_RIGHT)) {
              yyerror("Type error in power expression");
          } else {
              $$ = create_legal(AST_PWR, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_POWER;
          }
      }
    | TOKEN_PWR_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA obligation TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_OBLIGATION)) {
              yyerror("Type error in power expression");
          } else {
              $$ = create_legal(AST_PWR, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_POWER;
          }
      }
    | TOKEN_PWR_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA prohibition TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_PROHIBITION)) {
              yyerror("Type error in power expression");
          } else {
              $$ = create_legal(AST_PWR, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_POWER;
          }
      }
    | TOKEN_PWR_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA privilege TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_PRIVILEGE)) {
              yyerror("Type error in power expression");
          } else {
              $$ = create_legal(AST_PWR, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_POWER;
          }
      }
    ;

liability:
      TOKEN_LIAB_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA claim_right TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_CLAIM_RIGHT)) {
              yyerror("Type error in liability expression");
          } else {
              $$ = create_legal(AST_LIAB, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_LIABILITY;
          }
      }
    | TOKEN_LIAB_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA obligation TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_OBLIGATION)) {
              yyerror("Type error in liability expression");
          } else {
              $$ = create_legal(AST_LIAB, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_LIABILITY;
          }
      }
    | TOKEN_LIAB_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA prohibition TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_PROHIBITION)) {
              yyerror("Type error in liability expression");
          } else {
              $$ = create_legal(AST_LIAB, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_LIABILITY;
          }
      }
    | TOKEN_LIAB_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA privilege TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_PRIVILEGE)) {
              yyerror("Type error in liability expression");
          } else {
              $$ = create_legal(AST_LIAB, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_LIABILITY;
          }
      }
    ;

disability:
      TOKEN_DIS_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA claim_right TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_CLAIM_RIGHT)) {
              yyerror("Type error in disability expression");
          } else {
              $$ = create_legal(AST_DIS, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_DISABILITY;
          }
      }
    | TOKEN_DIS_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA obligation TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_OBLIGATION)) {
              yyerror("Type error in disability expression");
          } else {
              $$ = create_legal(AST_DIS, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_DISABILITY;
          }
      }
    | TOKEN_DIS_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA prohibition TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_PROHIBITION)) {
              yyerror("Type error in disability expression");
          } else {
              $$ = create_legal(AST_DIS, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_DISABILITY;
          }
      }
    | TOKEN_DIS_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA privilege TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_PRIVILEGE)) {
              yyerror("Type error in disability expression");
          } else {
              $$ = create_legal(AST_DIS, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_DISABILITY;
          }
      }
    ;

immunity:
      TOKEN_IMM_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA claim_right TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_CLAIM_RIGHT)) {
              yyerror("Type error in immunity expression");
          } else {
              $$ = create_legal(AST_IMM, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_IMMUNITY;
          }
      }
    | TOKEN_IMM_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA obligation TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_OBLIGATION)) {
              yyerror("Type error in immunity expression");
          } else {
              $$ = create_legal(AST_IMM, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_IMMUNITY;
          }
      }
    | TOKEN_IMM_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA prohibition TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_PROHIBITION)) {
              yyerror("Type error in immunity expression");
          } else {
              $$ = create_legal(AST_IMM, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_IMMUNITY;
          }
      }
    | TOKEN_IMM_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA privilege TOKEN_CLOSE_PAREN { 
          if (!check_expression_types($2, TYPE_ASSET) || !check_expression_types($4, TYPE_ASSET) || !check_expression_types($6, TYPE_PRIVILEGE)) {
              yyerror("Type error in immunity expression");
          } else {
              $$ = create_legal(AST_IMM, create_binary_op(AST_COMMA, $2, create_binary_op(AST_COMMA, $4, $6))); 
              $$->var_type = TYPE_IMMUNITY;
          }
      }
    ;
query_expression:
      power { 
          if (!check_expression_types($1, TYPE_POWER)) {
              yyerror("Type mismatch in query expression: expected TYPE_POWER");
              YYERROR;
          } else {

              $$->var_type = TYPE_QUERY; 
              $$ = create_query($1);
          }
      }
    | liability { 
          if (!check_expression_types($1, TYPE_LIABILITY)) {
              yyerror("Type mismatch in query expression: expected TYPE_LIABILITY");
              YYERROR;
          } else {

              $$->var_type = TYPE_QUERY; 
              $$ = create_query($1);              
          }
      }
    | disability { 
          if (!check_expression_types($1, TYPE_DISABILITY)) {
              yyerror("Type mismatch in query expression: expected TYPE_DISABILITY");
              YYERROR;
          } else {
              $$ = create_query($1);
              $$->var_type = TYPE_QUERY; 
          }
      }
    | immunity { 
          if (!check_expression_types($1, TYPE_IMMUNITY)) {
              yyerror("Type mismatch in query expression: expected TYPE_IMMUNITY");
              YYERROR;
          } else {
              $$ = create_query($1);
              $$->var_type = TYPE_QUERY; 
          }
      }
    ;

asset_type:
      TOKEN_SERVICE { $$ = create_type_node("Service"); }
    | TOKEN_PROPERTY { $$ = create_type_node("Property"); }
    ;

asset_subtype:
      TOKEN_NM { $$ = create_type_node("NM"); }
    | TOKEN_M { $$ = create_type_node("M"); }
    | TOKEN_PLUS { $$ = create_type_node("+"); }
    | TOKEN_MINUS { $$ = create_type_node("-"); }
    ;

%%


