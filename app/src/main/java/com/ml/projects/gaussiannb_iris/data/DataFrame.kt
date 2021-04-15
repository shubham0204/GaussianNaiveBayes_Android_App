package com.ml.projects.gaussiannb_iris.data

import android.content.Context
import com.opencsv.CSVReader
import java.io.InputStreamReader

class DataFrame( context: Context , assetsFileName : String ) {

    private var featureColumnData = HashMap<String,ArrayList<Float>>()
    private var rawData : List<Array<String>>
    private var columnNames : Array<String>
    private lateinit var readCSVCallback : ReadCSVCallback

    var numFeatures = 0
    var numSamples = 0
    var numClasses = 0
    lateinit var featureColumns : Array<FeatureColumn>
    lateinit var labels : Array<String>

    init {
        val csvReader = CSVReader( InputStreamReader( context.assets.open( assetsFileName ) ) )
        rawData = csvReader.readAll()
        columnNames = getColumnNames()
        initTable()
    }

    fun readCSV( readCSVCallback: ReadCSVCallback) {
        this.readCSVCallback = readCSVCallback
        populateColumns()
    }

    fun convertStringArrayToFloatArray( strArray : Array<String> ) : FloatArray {
        val out = strArray.map { si -> si.toFloat() }.toFloatArray()
        // TODO : Check for exceptions
        return out
    }

    override fun toString(): String {
        return "DataFrame with $numFeatures feature(s) and $numSamples samples."
    }

    interface ReadCSVCallback {
        fun onCSVProcessed(processedData : Array<FeatureColumn>, labels : Array<String> )
    }

    private fun getColumnNames() : Array<String> {
        val columnNames = rawData[ 0 ]
        println( columnNames.contentToString() )
        return columnNames
    }

    private fun populateColumns() {
        val labels = ArrayList<String>()

        for ( strSample in rawData.subList( 1 , rawData.size ) ) {
            labels.add( strSample[ numFeatures ] )
            val floatSample = convertStringArrayToFloatArray( strSample.sliceArray( IntRange( 0 , numFeatures - 1 ) ) )
            floatSample.forEachIndexed { index, fl ->
                featureColumnData[ columnNames[ index ] ]!!.add( fl )
            }
        }

        val featureColumnsList = ArrayList<FeatureColumn>()
        for ( ( name , data ) in featureColumnData ) {
            featureColumnsList.add(
                FeatureColumn( name , data.toFloatArray() )
            )
        }
        featureColumns = featureColumnsList.toTypedArray()
        this.labels = labels.toTypedArray()
        numClasses = this.labels.distinct().size

        readCSVCallback.onCSVProcessed( featureColumns, this.labels )
    }

    private fun initTable() {
        numFeatures = rawData[ 0 ].size - 1
        numSamples = rawData.size
        for ( i in 0 until numFeatures ) {
            featureColumnData[ columnNames[ i ] ] = ArrayList()
        }
    }




}