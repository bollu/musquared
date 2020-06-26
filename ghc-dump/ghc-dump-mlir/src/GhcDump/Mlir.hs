{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ParallelListComp #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module GhcDump.Mlir
    ( Pretty(..)
    , module GhcDump.Mlir
    ) where

import GhcDump.Ast
import GhcDump.Util

import Data.Ratio
import qualified Data.Text as T
import qualified Data.ByteString.Char8 as BS
import Text.PrettyPrint.ANSI.Leijen

data PrettyOpts = PrettyOpts { showUniques    :: Bool
                             , showIdInfo     :: Bool
                             , showLetTypes   :: Bool
                             , showUnfoldings :: Bool
                             , numArgs :: Int
                             }

defaultPrettyOpts :: PrettyOpts
defaultPrettyOpts = PrettyOpts { showUniques    = False
                               , showIdInfo     = False
                               , showLetTypes   = False
                               , showUnfoldings = False
                               , numArgs = 0
                               }
-- orphan
instance Pretty T.Text where
    pretty = text . T.unpack

instance Pretty ExternalName where
    pretty n@ExternalName{} = "\"" <> pretty (externalModuleName n) <> "." <> text (T.unpack $ externalName n) <> "\""
    pretty ForeignCall = "\"" <> "<foreign>" <> "\""

instance Pretty ModuleName where
    pretty = text . T.unpack . getModuleName

instance Pretty Unique where
    pretty = text . show

instance Pretty BinderId where
    pretty (BinderId b) = pretty b

instance Pretty Binder where
    pretty = pprBinder defaultPrettyOpts

pprBinder :: PrettyOpts -> Binder -> Doc
pprBinder opts b
  | showUniques opts = (pretty $ "%uniquebinder-" <> binderUniqueName b) <>  "-" <> pretty (numArgs opts)
  | otherwise        = (pretty $ "%binder-" <> (binderName $ unBndr b)) <> "-" <> pretty (numArgs opts)

instance Pretty TyCon where
    pretty (TyCon t _) = text $ T.unpack t

pprRational :: Rational -> Doc
pprRational r = pretty (numerator r) <> "/" <> pretty (denominator r)

instance Pretty Lit where
    pretty (MachChar x) = "'" <> char x <> "'"
    pretty (MachStr x) = "\"" <> text (BS.unpack x) <> "\""
    pretty MachNullAddr = "nullAddr#"
    pretty (MachInt x) = pretty x 
    pretty (MachInt64 x) = pretty x
    pretty (MachWord x) = pretty x 
    pretty (MachWord64 x) = pretty x
    pretty (MachFloat x) = "FLOAT" <> parens (pprRational x)
    pretty (MachDouble x) = "DOUBLE" <> parens (pprRational x)
    pretty (MachLabel x) = "LABEL"<> parens (pretty x)
    pretty (LitInteger x) = pretty x

instance Pretty CoreStats where
    pretty c =
        "Core Size"
        <>braces (hsep [ "terms="<>int (csTerms c)
                       , "types="<>int (csTypes c)
                       , "cos="<>int (csCoercions c)
                       , "vbinds="<>int (csValBinds c)
                       , "jbinds="<>int (csJoinBinds c)
                       ])

pprIdInfo :: PrettyOpts -> IdInfo Binder Binder -> IdDetails -> Doc
pprIdInfo opts i d
  | not $ showIdInfo opts = empty
  | otherwise = comment $ "IdInfo:" <+> align doc
  where
    doc = sep $ punctuate ", "
          $ [ pretty d
            , "arity=" <> pretty (idiArity i)
            , "inline=" <> pretty (idiInlinePragma i)
            , "occ=" <> pretty (idiOccInfo i)
            , "str=" <> pretty (idiStrictnessSig i)
            , "dmd=" <> pretty (idiDemandSig i)
            , "call-arity=" <> pretty (idiCallArity i)
            , "unfolding=" <> pprUnfolding opts (idiUnfolding i)
            ] ++ (if idiIsOneShot i then ["one-shot"] else [])

pprUnfolding :: PrettyOpts -> Unfolding Binder Binder -> Doc
pprUnfolding _    NoUnfolding = "NoUnfolding"
pprUnfolding _    BootUnfolding = "BootUnfolding"
pprUnfolding _    OtherCon{} = "OtherCon"
pprUnfolding _    DFunUnfolding = "DFunUnfolding"
pprUnfolding opts CoreUnfolding{..}
  | showUnfoldings opts = "CoreUnf" <> braces
     (align $ sep [ "is-value=" <> pretty unfIsValue
                  , "con-like=" <> pretty unfIsConLike
                  , "work-free=" <> pretty unfIsWorkFree
                  , "guidance=" <> pretty unfGuidance
                  , "template=" <> pprExpr opts unfTemplate
                  ])
  | otherwise = "CoreUnf{..}"

instance Pretty OccInfo where
    pretty OccManyOccs = "Many"
    pretty OccDead = "Dead"
    pretty OccOneOcc = "One"
    pretty (OccLoopBreaker strong) =
        if strong then "Strong Loopbrk" else "Weak Loopbrk"

instance Pretty IdDetails where
    pretty = text . show

data TyPrec   -- See Note [Precedence in types] in TyCoRep.hs
  = TopPrec         -- No parens
  | FunPrec         -- Function args; no parens for tycon apps
  | TyOpPrec        -- Infix operator
  | TyConPrec       -- Tycon args; no parens for atomic
  deriving( Eq, Ord )

pprType :: PrettyOpts -> Type -> Doc
pprType opts = pprType' opts TopPrec

pprType' :: PrettyOpts -> TyPrec -> Type -> Doc
pprType' opts _ (VarTy b)         = "core.VarTY-" <> pprBinder opts b
pprType' opts p t@(FunTy _ _)     = "core.FUNTY-" <> (maybeParens (p >= FunPrec) $ sep $ punctuate " ->" (map (pprType' opts FunPrec) (splitFunTys t)))
pprType' opts p (TyConApp tc [])  = (pprDebugDoubleQuote "core.TYCONAPP") <+>  "()" <+> "{ value = " <+> pprDebugDoubleQuote (pretty tc) <+> "}"  <+> ":" <+> pprDebugCoreAppRetty
pprType' opts p (TyConApp tc tys) = "core.TYCONAPP-" <> (maybeParens (p >= TyConPrec) $ pretty tc <+> hsep (map (pprType' opts TyConPrec) tys))
pprType' opts p (AppTy a b)       = "core.APPLYTY=" <> (maybeParens (p >= TyConPrec) $ pprType' opts TyConPrec a <+> pprType' opts TyConPrec b)
pprType' opts p t@(ForAllTy _ _)  = let (bs, t') = splitForAlls t
                                    in maybeParens (p >= TyOpPrec)
                                       $ "forall" <+> hsep (map (pprBinder opts) bs) <> "." <+> pprType opts t'
pprType' opts _ LitTy             = "LIT"
pprType' opts _ CoercionTy        = "Co"

maybeParens :: Bool -> Doc -> Doc
maybeParens True  = parens
maybeParens False = id

instance Pretty Type where
    pretty = pprType defaultPrettyOpts


pprDebugDoubleQuote :: Doc -> Doc
pprDebugDoubleQuote d = "\"" <> d <> "\""

pprDebugName :: String -> Doc -> Doc
pprDebugName name d = pprDebugDoubleQuote (pretty name) <+> "(" <> d <> ")"

pprExpr :: PrettyOpts -> Expr -> Doc
pprExpr opts = pprExpr' opts False


pprDebugRegionVarList :: Int -> [Doc] -> Doc
pprDebugRegionVarList start ds = hsep ["%arg-" <> pretty ix <> " = " <> d | d <- ds | ix <- [start,start+1..] :: [Int]]

pprDebugCoreAppRetty :: Doc
pprDebugCoreAppRetty = "() -> (!core.untyped)"

updOpts :: Int -> PrettyOpts -> PrettyOpts
updOpts n opts = opts { numArgs = (numArgs opts) + n }

pprDebugArgList :: Int -> [Doc] -> Doc
pprDebugArgList start ds = hsep $ punctuate "," ["%arg-" <> pretty ix <> " = " <> d | d <- ds | ix <- [start,start+1..] :: [Int]]


stubSubSubStubStubStub :: Doc
stubSubSubStubStubStub =  "\"core.stub\" () : () -> (!core.untyped)"

pprExpr' :: PrettyOpts -> Bool -> Expr -> Doc
pprExpr' opts _parens (EVar v)         = (pprDebugDoubleQuote "core.EVar") <+>  "()" <+> "{ value = " <> pprBinder opts v <> "}" <+> ":" <+> pprDebugCoreAppRetty
-- pprExpr' opts _parens (EVar v)         = pprDebugName "EVar" $ pprBinder opts v
pprExpr' opts _parens (EVarGlobal v)   = (pprDebugDoubleQuote "EvarGlobal") <+> "()" <+> "{value = " <> pretty v <> "}" <+> ":" <+> pprDebugCoreAppRetty

-- pprExpr' opts _parens (EVarGlobal v)   = (pprDebugName "core.EvarGlobal") <+> "()" <+> "{ value = " <> pretty v <> "}" <+> ":" <+> pprDebugCoreAppRetty
pprExpr' opts _parens (ELit l)         = (pprDebugDoubleQuote "core.ELit") <+> "()" <+> "{ value = " <> pretty l <> "} " <+> ":" <+> pprDebugCoreAppRetty
pprExpr' opts parens  e@(EApp{})       = 
    pprDebugDoubleQuote "core.EApp" <+> "()" <+> "({" <> (pprDebugRegionVarList (numArgs opts) $ [pprArg ix y | y <- ys | ix <- [0, 1..]]) <+> "})" <+> ":" <+> pprDebugCoreAppRetty
    where
        (x, ys) = collectArgs e
        cumlen = [100, 200..]
        pprArg ix (EType t) = pprType' (updOpts (cumlen !! ix) opts) TyConPrec t
        pprArg ix x         = pprExpr' (updOpts (cumlen !! ix) opts) True x


-- let (x, ys) = collectArgs e
--                                          in pprDebugDoubleQuote "core.EApp" <+> "()" <+> "({" <> (pprDebugRegionVarList (numArgs opts) $ map pprArg ys) <+> "})" <+> ":" <+> pprDebugCoreAppRetty
--   where pprArg (EType t) = char '@' <> pprType' opts' TyConPrec t
--         pprArg x         = pprExpr' opts True x
--         opts' = updOpts (length ys) opts
pprExpr' opts parens  x@(ETyLam _ _)   = let (bs, x') =  collectTyBinders x
                                         in pprDebugName "ETyLam" $ maybeParens parens
                                                                    $ hang' ("Λ" <+> sep (map (pprBinder opts) bs) <+> smallRArrow) 2 (pprExpr' opts False x')

-- pprExpr' opts parens  x@(ELam _ _)     =  (pprDebugDoubleQuote "ELam") <+> "()" <+> ":" <+> pprDebugCoreAppRetty


pprExpr' opts parens  x@(ELam _ _)     = let (bs, x') = collectBinders x
                                         in (pprDebugDoubleQuote "core.ELam") <+> "()" <+> 
                                            "({" <>  
                                            "^entry(" <> 
                                            (hsep  $ punctuate "," [pprBinder opts b <> ":" <> pprDebugCoreAppRetty | b <- bs]) <> 
                                            "):" 
                                            <+> stubSubSubStubStubStub -- (pprExpr' opts False x') 
                                            <+> "})" <+> ":" <+> pprDebugCoreAppRetty

pprExpr' opts parens  (ELet xs y)      = pprDebugName "ELet" $ maybeParens parens $ "let" <+> (align $ vcat $ map (uncurry (pprBinding opts)) xs)
                                         <$$> "in" <+> align (pprExpr' opts False y)
  where pprBind (b, rhs) = pprBinder opts b <+> equals <+> align (pprExpr' opts False rhs)
pprExpr' opts parens  (ECase x b alts) = pprDebugName "ECase" $ maybeParens parens
                                         $ sep [ sep [ "case" <+> pprExpr' opts False x
                                                     , "of" <+> pprBinder opts b <+> "{" ]
                                               , indent 2 $ vcat $ map pprAlt alts
                                               , "}"
                                               ]
  where pprAlt (Alt con bndrs rhs) = pprDebugName "Alt" $ hang' (hsep (pretty con : map (pprBinder opts) bndrs) <+> smallRArrow) 2 (pprExpr' opts False rhs)
pprExpr' opts parens  (EType t)        = pprDebugName "EType" $  maybeParens parens $ "TYPE:" <+> pprType opts t
pprExpr' opts parens  ECoercion        = pprDebugName "ECoercion" $ "CO"

instance Pretty AltCon where
    pretty (AltDataCon t) = text $ T.unpack t
    pretty (AltLit l) = pretty l
    pretty AltDefault = text "DEFAULT"

instance Pretty Expr where
    pretty = pprExpr defaultPrettyOpts

pprTopBinding :: PrettyOpts -> TopBinding -> Doc
pprTopBinding opts tb =
    case tb of
      NonRecTopBinding b s rhs -> pprTopBind (b,s,rhs)
      RecTopBinding bs -> "rec" <+> braces (line <> vsep (map pprTopBind bs))
  where
    pprTopBind (b@(Bndr b'),s,rhs) =
        pprTypeSig opts b
        <$$> pprIdInfo opts (binderIdInfo b') (binderIdDetails b')
        <$$> comment (pretty s)
        <$$> hang' (pprBinder opts b <+> equals) 2 (pprExpr opts rhs)
        <> line

prettyDebugComment :: Doc -> Doc
prettyDebugComment d = "//" <+> d

pprTypeSig :: PrettyOpts -> Binder -> Doc
pprTypeSig opts b@(Bndr b') =
    prettyDebugComment $ pprBinder opts b <+> dcolon <+> align (pprType opts (binderType b'))

{-
pprBinding :: PrettyOpts -> Binder -> Expr -> Doc
pprBinding opts b@(Bndr b'@Binder{}) rhs =
    ppWhen (showLetTypes opts) (pprTypeSig opts b)
    <$$> pprIdInfo opts (binderIdInfo b') (binderIdDetails b')
    <$$> hang' (pprBinder opts b <+> equals) 2 (pprExpr opts rhs)
pprBinding opts b@(Bndr TyBinder{}) rhs =
    -- let-bound type variables: who knew?
    hang' (pprBinder opts b <+> equals) 2 (pprExpr opts rhs)
-}

pprBinding :: PrettyOpts -> Binder -> Expr -> Doc
pprBinding opts b@(Bndr b'@Binder{}) rhs =
    ppWhen (showLetTypes opts) (pprTypeSig opts b)
    <$$> pprIdInfo opts (binderIdInfo b') (binderIdDetails b')
    <$$> hang' (pprBinder opts b <+> equals) 2 (pprExpr opts rhs)
pprBinding opts b@(Bndr TyBinder{}) rhs =
    -- let-bound type variables: who knew?
    hang' (pprBinder opts b <+> equals) 2 (pprExpr opts rhs)

instance Pretty TopBinding where
    pretty = pprTopBinding defaultPrettyOpts


pprTopBindingMlir :: PrettyOpts -> TopBinding -> Doc
pprTopBindingMlir opts tb = 
    case tb of
      NonRecTopBinding b s rhs ->  pprTopBind (b, s, rhs)
      RecTopBinding bs -> line <> vsep (map pprTopBind bs)
      -- RecTopBinding bs -> pprDebugName "REC" $ 
      --                           braces (line <> vsep (map pprTopBind bs))
  where
    pprTopBind (b@(Bndr b'),s,rhs) =
        pprTypeSig opts b
        <$$> pprIdInfo opts (binderIdInfo b') (binderIdDetails b')
        <$$> comment (pretty s)
        <$$> hang' (pprBinder opts b <+> equals) 2 (pprExpr opts rhs)
        <> line


pprModule :: PrettyOpts -> Module -> Doc
pprModule opts m =
    comment ("MLIR")
    <$$> comment (pretty $ modulePhase m)
    <$$> text "module" <+> ("@" <> pretty (moduleName m)) <+> "{" <> line
    -- <$$> vsep (map (pprTopBinding opts) (moduleTopBindings m)) 
    <$$> vsep ([pprTopBindingMlir (updOpts ix opts) bind | bind <- moduleTopBindings m | ix <- [1001, 2001..]])
    <$$> text "}"

instance Pretty Module where
    pretty = pprModule defaultPrettyOpts

comment :: Doc -> Doc
-- comment x = "{-" <+> x <+> "-}"
comment x = "//" <+> x

dcolon :: Doc
dcolon = "::"

smallRArrow :: Doc
smallRArrow = "→"

hang' :: Doc -> Int -> Doc -> Doc
hang' d1 n d2 = hang n $ sep [d1, d2]

ppWhen :: Bool -> Doc -> Doc
ppWhen True x = x
ppWhen False _ = empty
