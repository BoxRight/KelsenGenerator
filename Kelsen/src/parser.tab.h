/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_PARSER_TAB_H_INCLUDED
# define YY_YY_PARSER_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    TOKEN_INT_LITERAL = 258,       /* TOKEN_INT_LITERAL  */
    TOKEN_FLOAT_LITERAL = 259,     /* TOKEN_FLOAT_LITERAL  */
    TOKEN_STRING_LITERAL = 260,    /* TOKEN_STRING_LITERAL  */
    TOKEN_IDENTIFIER = 261,        /* TOKEN_IDENTIFIER  */
    TOKEN_TYPE_INT = 262,          /* TOKEN_TYPE_INT  */
    TOKEN_TYPE_FLOAT = 263,        /* TOKEN_TYPE_FLOAT  */
    TOKEN_TYPE_STRING = 264,       /* TOKEN_TYPE_STRING  */
    TOKEN_TYPE_ASSET = 265,        /* TOKEN_TYPE_ASSET  */
    TOKEN_TYPE_SUBJECT = 266,      /* TOKEN_TYPE_SUBJECT  */
    TOKEN_TYPE_CLAUSE = 267,       /* TOKEN_TYPE_CLAUSE  */
    TOKEN_TYPE_QUERY = 268,        /* TOKEN_TYPE_QUERY  */
    TOKEN_ADD = 269,               /* TOKEN_ADD  */
    TOKEN_SUB = 270,               /* TOKEN_SUB  */
    TOKEN_PLUS = 271,              /* TOKEN_PLUS  */
    TOKEN_MINUS = 272,             /* TOKEN_MINUS  */
    TOKEN_ASSIGN = 273,            /* TOKEN_ASSIGN  */
    TOKEN_SEMICOLON = 274,         /* TOKEN_SEMICOLON  */
    TOKEN_COMMA = 275,             /* TOKEN_COMMA  */
    TOKEN_CR_OPEN = 276,           /* TOKEN_CR_OPEN  */
    TOKEN_OB_OPEN = 277,           /* TOKEN_OB_OPEN  */
    TOKEN_PR_OPEN = 278,           /* TOKEN_PR_OPEN  */
    TOKEN_PVG_OPEN = 279,          /* TOKEN_PVG_OPEN  */
    TOKEN_CLOSE_PAREN = 280,       /* TOKEN_CLOSE_PAREN  */
    TOKEN_OPEN_BRACE = 281,        /* TOKEN_OPEN_BRACE  */
    TOKEN_CLOSE_BRACE = 282,       /* TOKEN_CLOSE_BRACE  */
    TOKEN_CONDITION = 283,         /* TOKEN_CONDITION  */
    TOKEN_CONSEQUENCE = 284,       /* TOKEN_CONSEQUENCE  */
    TOKEN_AND = 285,               /* TOKEN_AND  */
    TOKEN_PWR_OPEN = 286,          /* TOKEN_PWR_OPEN  */
    TOKEN_LIAB_OPEN = 287,         /* TOKEN_LIAB_OPEN  */
    TOKEN_DIS_OPEN = 288,          /* TOKEN_DIS_OPEN  */
    TOKEN_IMM_OPEN = 289,          /* TOKEN_IMM_OPEN  */
    TOKEN_UNKNOWN = 290,           /* TOKEN_UNKNOWN  */
    TOKEN_SERVICE = 291,           /* TOKEN_SERVICE  */
    TOKEN_PROPERTY = 292,          /* TOKEN_PROPERTY  */
    TOKEN_NM = 293,                /* TOKEN_NM  */
    TOKEN_M = 294                  /* TOKEN_M  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 26 "parser.y"

    int ival;
    double fval;
    char *sval;
    struct ASTNode *node;

#line 110 "parser.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_PARSER_TAB_H_INCLUDED  */
