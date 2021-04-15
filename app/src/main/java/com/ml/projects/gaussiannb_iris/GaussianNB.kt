package com.ml.projects.gaussiannb_iris

import com.ml.projects.gaussiannb_iris.data.DataFrame
import kotlin.math.exp
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt

class GaussianNB( private var dataFrame : DataFrame ) {

    private var featureDistributions : Array<GaussianDistribution>
    private var priorProbabilities : HashMap<String,Float>
    private var resultCallback : ResultCallback? = null

    init {

        val GDs = ArrayList<GaussianDistribution>()
        for ( featureColumn in dataFrame.featureColumns ) {
            GDs.add( GaussianDistribution( featureColumn.featureMean , featureColumn.featureStdDev ))
        }
        featureDistributions = GDs.toTypedArray()

        priorProbabilities = computePriorProbabilities( dataFrame.labels )
        println( priorProbabilities.toString() )

        println( predictLabel( floatArrayOf( 5.1f , 3.5f , 1.4f , 0.2f )) )

    }

    fun predict( x : FloatArray , resultCallback: ResultCallback ) {
        this.resultCallback = resultCallback
        predictLabel( x )
    }

    interface ResultCallback {
        fun onPredictionResult( label : String , probs : FloatArray )
    }

    private fun predictLabel( sample : FloatArray ) {
        val probArray = FloatArray( dataFrame.numClasses )
        for ( ( i , priorProb ) in priorProbabilities.values.withIndex()) {
            var p = log10( priorProb )
            for ( j in 0 until dataFrame.numFeatures ) {
                p += featureDistributions[ j ].getLogProb( sample[ i ] )
            }
            probArray[ i ] = p
        }
        val label = priorProbabilities.keys.toTypedArray()[ probArray.indexOf( probArray.max()!!) ]
        resultCallback?.onPredictionResult( label , probArray )
    }

    private fun computePriorProbabilities( labels : Array<String> ) : HashMap<String,Float> {
        val labelCountMap = labels.groupingBy { it }.eachCount()
        val out = HashMap<String,Float>()
        for ( ( label , count ) in labelCountMap ) {
            out[ label ] = count.toFloat() / dataFrame.numSamples.toFloat()
        }
        return out
    }


    class GaussianDistribution( var mean : Float , var stdDev : Float ) {

        private val p = 1f / ( sqrt( 2.0 * Math.PI ) ).toFloat()

        fun getProb( x : Float ) : Float {
            val exp = exp( -0.5 * ((x - mean) / stdDev ).pow( 2 ) ).toFloat()
            return ( 1 / stdDev ) * p * exp
        }

        fun getLogProb( x : Float ) : Float {
            return log10( getProb( x ) )
        }

    }




}