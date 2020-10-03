import qualified Data.Map

data Tensor a = Scalar a (VJP a)
type VJP a = Tensor a -> [(String, Tensor a)]

tag :: String -> Tensor a -> Tensor a
tag name (Scalar x vjp) = Scalar x (\g -> (name, g) : vjp g)

grad :: Num a => [String] -> Tensor a -> [Tensor a]
grad keys (Scalar _ vjp) = mapM lookup_exn keys accumulate_keys
  where accumulate_keys = Data.Map.fromListWith ( + ) (vjp 1)
        lookup_exn key map = let (Just ret) = Data.Map.lookup key map in ret

stopgrad :: Tensor a -> Tensor a
stopgrad (Scalar x _) = Scalar x (\g -> [])
        
instance Eq a => Eq (Tensor a) where
  (Scalar a _) == (Scalar b _) = a == b

instance Ord a => Ord (Tensor a) where
  compare (Scalar a _) (Scalar b _) = compare a b

instance Show a => Show (Tensor a) where
  show (Scalar x _) = show x

instance Num a => Num (Tensor a) where
  sa@(Scalar a vjpa) * sb@(Scalar b vjpb) = Scalar (a * b) (\g -> vjpa (g * sb) ++ vjpb (g * sa))
  (Scalar a vjpa) + (Scalar b vjpb) = Scalar (a + b) (\g -> vjpa g ++ vjpb g)
  abs sa@(Scalar a vjp) = Scalar (abs a) (\g -> vjp (signum sa * g))
  fromInteger a = Scalar (fromInteger a) (\g -> [])
  negate (Scalar a vjp) = Scalar (negate a) (\g -> vjp (negate g))
  signum (Scalar a vjp) = Scalar (signum a) (\g -> vjp 0)

instance Fractional a => Fractional (Tensor a) where
  recip sa@(Scalar a vjp) = Scalar (recip a) (\g -> vjp (-g/(sa*sa)))
  fromRational a = Scalar (fromRational a) (\g -> [])

instance Floating a => Floating (Tensor a) where
  pi = Scalar pi (\g -> [])
  exp sa@(Scalar a vjp) = Scalar (exp a) (\g -> vjp (g * exp sa))
  log sa@(Scalar a vjp) = Scalar (log a) (\g -> vjp (g / sa))
  sin sa@(Scalar a vjp) = Scalar (sin a) (\g -> vjp (g * cos sa))
  cos sa@(Scalar a vjp) = Scalar (cos a) (\g -> vjp (-g * sin sa))
  asin sa@(Scalar a vjp) = Scalar (asin a) (\g -> vjp (g / sqrt (1 - sa * sa)))
  acos sa@(Scalar a vjp) = Scalar (acos a) (\g -> vjp (-g / sqrt (1 - sa * sa)))
  atan sa@(Scalar a vjp) = Scalar (atan a) (\g -> vjp (g / (sa * sa + 1)))
  sinh sa@(Scalar a vjp) = Scalar (sinh a) (\g -> vjp (g * cosh sa))
  cosh sa@(Scalar a vjp) = Scalar (cosh a) (\g -> vjp (g * sinh sa))
  asinh sa@(Scalar a vjp) = Scalar (asinh a) (\g -> vjp (g / sqrt (sa * sa + 1)))
  acosh sa@(Scalar a vjp) = Scalar (acosh a) (\g -> vjp (g / sqrt (sa * sa - 1)))
  atanh sa@(Scalar a vjp) = Scalar (atanh a) (\g -> vjp (g / (1 - sa * sa)))

main :: IO ()
main =
  let x = tag "x" 4
      f = x ^ 2 + stopgrad (3 * x)
      [f'] = grad ["x"] f
      [f''] = grad ["x"] f'
  in do putStrLn $ "x = " ++ show x
        putStrLn $ "f(x) = " ++ show f
        putStrLn $ "f'(x) = " ++ show f'
        putStrLn $ "f''(x) = " ++ show f''
