package com.ml.projects.gaussiannb_iris

import com.ml.projects.gaussiannb_iris.data.DataFrame
import kotlin.math.exp
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt

// Class to implement Gaussian Naive Bayes
class GaussianNB( private var dataFrame : DataFrame ) {

    // Array to store Gaussian Distributions for each feature.
    private var featureDistributions : Array<GaussianDistribution>

    // Prior probabilities stored in a HashMap of form ( column_name , prior_prob )
    private var priorProbabilities : HashMap<String,Float>

    private var resultCallback : ResultCallback? = null

    init {

        // Construct Gaussian Distributions for each feature and append them to `featureDistributions`.
        val GDs = ArrayList<GaussianDistribution>()
        for ( featureColumn in dataFrame.featureColumns ) {
            GDs.add( GaussianDistribution( featureColumn.featureMean , featureColumn.featureStdDev ))
        }
        featureDistributions = GDs.toTypedArray()

        // Compute prior probabilities and store them in `priorProbabilities`.
        priorProbabilities = computePriorProbabilities( dataFrame.labels )

    }

    // Predict the label for the given sample.
    fun predict( x : FloatArray , resultCallback: ResultCallback ) {
        this.resultCallback = resultCallback
        predictLabel( x )
    }

    // Callback to check whether the calculations are done.
    interface ResultCallback {
        fun onPredictionResult( label : String , probs : FloatArray )
    }

    // Predict the label for the given sample.
    private fun predictLabel( sample : FloatArray ) {
        val probArray = FloatArray( dataFrame.numClasses )
        for ( ( i , priorProb ) in priorProbabilities.values.withIndex()) {
            // We take the log probabilities so as to avoid underflow.
            var p = log10( priorProb )
            // While we take log, the product is transformed into a sum
            // log( a . b ) = log(a) + log(b)
            for ( j in 0 until dataFrame.numFeatures ) {
                p += featureDistributions[ j ].getLogProb( sample[ i ] )
            }
            probArray[ i ] = p
        }
        // Get the label with highest probability.
        val label = priorProbabilities.keys.toTypedArray()[ probArray.indexOf( probArray.max()!!) ]
        resultCallback?.onPredictionResult( label , probArray )
    }

    // Compute the prior probabilities.
    // These probabilities are p( class=some_class ) which are calculated as
    // p( class=apple ) = num_samples_label_as_apple / num_samples_in_ds
    private fun computePriorProbabilities( labels : Array<String> ) : HashMap<String,Float> {
        // Get the count ( freq ) of each unique class in labels.
        val labelCountMap = labels.groupingBy { it }.eachCount()
        // The prior probabilties are stored in a HashMap of form ( column_name , prob )
        val out = HashMap<String,Float>()
        for ( ( label , count ) in labelCountMap ) {
            // Append the prob with key=column_name
            out[ label ] = count.toFloat() / dataFrame.numSamples.toFloat()
        }
        return out
    }

    // A Gaussian Distribution which will be constructed for every feature, given the mean and standard deviation.
    class GaussianDistribution( var mean : Float , var stdDev : Float ) {

        private val p = 1f / ( sqrt( 2.0 * Math.PI ) ).toFloat()

        // Get the likelihood for given x.
        fun getProb( x : Float ) : Float {
            val exp = exp( -0.5 * ((x - mean) / stdDev ).pow( 2 ) ).toFloat()
            return ( 1 / stdDev ) * p * exp
        }

        // Get the log of the likelihood for given x.
        fun getLogProb( x : Float ) : Float {
            return log10( getProb( x ) )
        }

    }




}