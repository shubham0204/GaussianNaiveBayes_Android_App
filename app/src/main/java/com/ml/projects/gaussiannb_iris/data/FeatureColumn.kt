package com.ml.projects.gaussiannb_iris.data

import kotlin.math.pow

// Holds data for a particular feature.
class FeatureColumn( var name : String , var data : FloatArray ) {

    // Mean of given `data`
    var featureMean : Float

    // Variance of given `data`
    var featureVariance : Float

    // Standard deviation of given `data`
    var featureStdDev : Float

    init {

        featureMean = computeMean()
        featureVariance = computeVariance()

        // Compute the standard deviation of `data` = sqrt( variance )
        featureStdDev = featureVariance.pow( 0.5f )
    }

    override fun toString(): String {
        return "{ name = $name , mean = $featureMean , stddev = $featureStdDev }"
    }

    // Compute the mean of `data`.
    private fun computeMean() : Float {
        return data.average().toFloat()
    }

    // Compute the variance of `data`.
    private fun computeVariance() : Float {
        val mean = data.average().toFloat()
        val meanSquaredSum = data.map{ xi -> ( xi - mean ).pow( 2 ) }.sum()
        return meanSquaredSum / data.size
    }

}