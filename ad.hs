import qualified Data.Map

data Tensor a = Tensor {
  val :: a,
  vjp :: Tensor a -> [(String, Tensor a)]
}

tag :: String -> Tensor a -> Tensor a
tag name tensor = tensor { vjp = \g -> (name, g) : filter ((/= name) . fst) (vjp tensor g) }

grad :: Num a => [String] -> Tensor a -> [Tensor a]
grad keys tensor = mapM lookup_exn keys accumulate_keys
  where accumulate_keys = Data.Map.fromListWith ( + ) (vjp tensor 1)
        lookup_exn key map = let (Just ret) = Data.Map.lookup key map in ret

stopgrad :: Tensor a -> Tensor a
stopgrad tensor = tensor { vjp = const [] }
        
instance Eq a => Eq (Tensor a) where
  ta == tb = val ta == val tb

instance Ord a => Ord (Tensor a) where
  compare ta tb = compare (val ta) (val tb)

instance Show a => Show (Tensor a) where
  show = show . val

instance Num a => Num (Tensor a) where
  ta * tb = Tensor { val = val ta * val tb, vjp = \g -> vjp ta (g * tb) ++ vjp tb (g * ta) }
  ta + tb = Tensor { val = val ta + val tb, vjp = \g -> vjp ta g ++ vjp tb g }
  abs ta = Tensor { val = abs (val ta), vjp = vjp ta . (* signum ta) }
  fromInteger v = Tensor { val = fromInteger v, vjp = const [] }
  negate ta = Tensor { val = negate (val ta), vjp = vjp ta . negate }
  signum ta = Tensor { val = signum (val ta), vjp = vjp ta . const 0 }

instance Fractional a => Fractional (Tensor a) where
  recip ta = Tensor { val = recip (val ta), vjp = vjp ta . (/ negate (ta * ta)) }
  fromRational v = Tensor { val = fromRational v, vjp = const [] }

instance Floating a => Floating (Tensor a) where
  pi = Tensor { val = pi, vjp = const [] }
  exp ta = Tensor { val = exp (val ta), vjp = vjp ta . (* exp ta) }
  log ta = Tensor { val = log (val ta), vjp = vjp ta . (/ ta) }
  sin ta = Tensor { val = sin (val ta), vjp = vjp ta . (* cos ta) }
  cos ta = Tensor { val = cos (val ta), vjp = vjp ta . (* negate (sin ta)) }
  asin ta = Tensor { val = asin (val ta), vjp = vjp ta . (/ sqrt (1 - ta * ta)) }
  acos ta = Tensor { val = acos (val ta), vjp = vjp ta . (/ negate (sqrt (1 - ta * ta))) }
  atan ta = Tensor { val = atan (val ta), vjp = vjp ta . (/ (ta * ta + 1)) }
  sinh ta = Tensor { val = sinh (val ta), vjp = vjp ta . (* cosh ta) }
  cosh ta = Tensor { val = cosh (val ta), vjp = vjp ta . (* sinh ta) }
  asinh ta = Tensor { val = asinh (val ta), vjp = vjp ta . (/ sqrt (ta * ta + 1)) }
  acosh ta = Tensor { val = acosh (val ta), vjp = vjp ta . (/ sqrt (ta * ta - 1)) }
  atanh ta = Tensor { val = atanh (val ta), vjp = vjp ta . (/ (1 - ta * ta)) }

d_dx :: Num a => (Tensor a -> Tensor a) -> Tensor a -> Tensor a
d_dx f x = head . grad ["x"] . f $ tag "x" x

main :: IO ()
main =
  let f x = x ^ 2 + stopgrad (3 * x)
      f'  = d_dx f
      f'' = d_dx f'
      x = 4
  in do putStrLn $ "x = " ++ show x
        putStrLn $ "f(x) = " ++ show (f x)
        putStrLn $ "f'(x) = " ++ show (f' x)
        putStrLn $ "f''(x) = " ++ show (f'' x)
