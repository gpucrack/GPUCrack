# How to hash passwords

Very easy to use, first you'll need an array of type Password. <br><br>
Then you'll need to check the length of your passwords, if it's 6 for example, modify the value "PASSWORD_LENGTH" inside
constants.cuh. <br> <br>
Once these things are done, you're good to go. If you want to hash without using a time variable to retrieve execution
time type: <br>
<code>auto * result = parallelized_hash(passwords, passwordNumber);
</code> <br>
All the digests will be retrieved inside result, you don't even need to malloc it. <br> <br>
Don't forget to free the memory ! Function is not handling it ! <br> Type: <br>
<code>free(passwords); <br>
free(result);</code> <br>
Where passwords is the array that you created before. <br> <br>
If you want to use a time variable, you can pass it to the function as a pointer: <br>
<code>auto * result = parallelized_hash(passwords, passwordNumber, &timeVariable);
</code> <br><br>
<strong>timeVariable MUST be of type float.</strong>

# Testing results

If you modify any part of this code, you'll need to test it. <br><br>
For compliance (a.k.a the result is what you expected): <br>
Just run complianceTest executable.<br><br>

For a general test (a.k.a runs in a normal scenario): <br>
Just run generalTest executable.<br><br>

For performance (a.k.a find the best Thread/Block value for performance): <br>
Just run benchmarkTest executable.<br><br>