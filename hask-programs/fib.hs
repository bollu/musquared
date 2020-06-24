{-# LANGUAGE NoImplicitPrelude #-}
fib :: Int -> Int
fib i = case i of
          0 -> 0
          1 -> 1
          n -> fib n + fib (n - 1)

main :: IO ()
main = putStrLn (show (fib 10))
