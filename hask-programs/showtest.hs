showint = show 1
showstr = show "foo"
showfloat = show (42.0)

main :: IO ()
main = print (showint) >> print (showstr) >> print (showfloat)
  

