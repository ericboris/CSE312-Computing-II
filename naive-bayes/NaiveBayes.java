// Do not submit with package statements if you are using eclipse.
// Only use what is provided in the standard libraries.

import java.io.*;
import java.util.*;

public class NaiveBayes {
    private HashMap<String, Double> logProbGivenHamMap = new HashMap<String, Double>();
    private HashMap<String, Double> logProbGivenSpamMap = new HashMap<String, Double>();

    private double logProbOfHam;
    private double logProbOfSpam;

    /*
     * !! DO NOT CHANGE METHOD HEADER !!
     * If you change the method header here, our grading script won't
     * work and you will lose points!
     * 
     * Train your Naive Bayes Classifier based on the given training
     * ham and spam emails.
     *
     * Params:
     *      hams - email files labeled as 'ham'
     *      spams - email files labeled as 'spam'
     */
    public void train(File[] hams, File[] spams) throws IOException {
        // Calculate the log probabilities of Ham or Spam
        logProbOfHam = Math.log(hams.length / (double) (hams.length + spams.length));
        logProbOfSpam = Math.log(spams.length / (double) (hams.length + spams.length));

        // Create maps to hold the cardinality of each word appearing in the training data
        HashMap<String, Integer> hamWordsCardinality = new HashMap<>();
        HashMap<String, Integer> spamWordsCardinality = new HashMap<>();

        // Fill the maps with the cardinality of words appearing in the training data
        readFilesToMap(hams, hamWordsCardinality);
        readFilesToMap(spams, spamWordsCardinality);

        // Include any words not in one training set as zeroes in the other
        fillMissingKeys(hamWordsCardinality, spamWordsCardinality);
        fillMissingKeys(spamWordsCardinality, hamWordsCardinality);

        // Calculate the log probabilities of those words appearing in the training data given the sample space
        calcLogProbXGivenY(hamWordsCardinality, logProbGivenHamMap, hams.length);
        calcLogProbXGivenY(spamWordsCardinality, logProbGivenSpamMap, spams.length);
    }

    /**
     * Get the set of words in each file and tally their sum across files
     *
     * @param files an array of files to iterate over
     * @param map holds the results with words as keys and word counts as values
     * @throws IOException
     */
    private void readFilesToMap(File[] files, HashMap<String, Integer> map) throws IOException {
        for (File f : files) {
            Set<String> tokenSet = tokenSet(f);
            for (String s : tokenSet) {
                incrementMapValue(map, s);
            }
        }
    }

    /**
     * Increment the key's value in map
     *
     * @param map the map to increment values in
     * @param key the key whose value to increment
     */
    private void incrementMapValue(Map<String, Integer> map, String key) {
        // containsKey() checks if this map contains a mapping for a key
        int count = map.getOrDefault(key, 0);
        map.put(key, count + 1);
    }

    /**
     * Fill map a with any elements it's missing from b and set the added values to zero
     *
     * @param a the map to fill with elements not previously included
     * @param b the map to draw un-included elements from
     */
    private void fillMissingKeys(Map<String, Integer> a, Map<String, Integer> b) {
        for (String key : b.keySet()) {
            if (!a.containsKey(key)) {
                a.put(key, 0);
            }
        }
    }

    /**
     * Calculate the log probability of a key appearing in a set given the total set size,
     * assign the log results to a map by key
     *
     * @param cardMap a map that holds the cardinality of keys appearances
     * @param logMap a map to hold the log probabilities of key appearances
     * @param cardinal the cardinality of the sample space
     */
    private void calcLogProbXGivenY(HashMap<String, Integer> cardMap, HashMap<String, Double> logMap, int cardinal) {
        for (String key : cardMap.keySet()) {
            // numerator = cardMap.get(key) + 1
            // denominator = (double) (cardinal + 2)
            // Cast denominator to double to insure double division
            // Apply Laplace Smoothing: increment numerator by 1 and denominator by 2
            // Take log of result to prevent underflow
            logMap.put(key, Math.log((cardMap.get(key) + 1) / (double) (cardinal + 2)));
        }
    }

    /*
     * !! DO NOT CHANGE METHOD HEADER !!
     * If you change the method header here, our grading script won't
     * work and you will lose points!
     *
     * Classify the given unlabeled set of emails. Add each email to the correct
     * label set. SpamFilterMain.java would follow the format in
     * example_output.txt and output your result to stdout. Note the order
     * of the emails in the output does NOT matter.
     * 
     *
     * Params:
     *      emails - unlabeled email files to be classified
     *      spams  - set for spam emails that needs to be populated
     *      hams   - set for ham emails that needs to be populated
     */
    public void classify(File[] emails, Set<File> spams, Set<File> hams) throws IOException {
        // Store the unique words appearing in each email
        HashSet<String> wordSet;

        // Look at each email in turn
        for (File e : emails) {
            // Store it's unique words
            wordSet = tokenSet(e);

            // Store the sum of the log probabilities of the words in the email given
            // the probability with which they appear in ham and spam emails
            double logProbWordGivenHam = 0d;
            double logProbWordGivenSpam = 0d;

            // Sum the log probabilities of words in the email
            for (String w : wordSet) {
                if (logProbGivenHamMap.containsKey(w)) {
                    logProbWordGivenHam += logProbGivenHamMap.get(w);
                }
                if (logProbGivenSpamMap.containsKey(w)) {
                    logProbWordGivenSpam += logProbGivenSpamMap.get(w);
                }
            }

            // Compare whether an email is more likely to be spam based on whether
            // the log probabilities of spam are higher than the log probabilities of ham
            // and email to the appropriate file array based on the decision
            if (logProbOfSpam + logProbWordGivenSpam > logProbOfHam + logProbWordGivenHam) {
                spams.add(e);
            } else {
                hams.add(e);
            }
        }
    }

    /*
     *  Helper Function:
     *  This function reads in a file and returns a set of all the tokens. 
     *  It ignores "Subject:" in the subject line.
     *  
     *  If the email had the following content:
     *  
     *  Subject: Get rid of your student loans
     *  Hi there ,
     *  If you work for us , we will give you money
     *  to repay your student loans . You will be 
     *  debt free !
     *  FakePerson_22393
     *  
     *  This function would return to you
     *  ['be', 'student', 'for', 'your', 'rid', 'we', 'of', 'free', 'you', 
     *   'us', 'Hi', 'give', '!', 'repay', 'will', 'loans', 'work', 
     *   'FakePerson_22393', ',', '.', 'money', 'Get', 'there', 'to', 'If', 
     *   'debt', 'You']
     */
    public static HashSet<String> tokenSet(File filename) throws IOException {
        HashSet<String> tokens = new HashSet<String>();
        Scanner filescan = new Scanner(filename);
        filescan.next(); // Ignoring "Subject"
        while (filescan.hasNextLine() && filescan.hasNext()) {
            tokens.add(filescan.next());
        }
        filescan.close();
        return tokens;
    }
}
