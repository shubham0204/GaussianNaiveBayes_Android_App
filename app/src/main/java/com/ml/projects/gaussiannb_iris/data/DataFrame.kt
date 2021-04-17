package com.ml.projects.gaussiannb_iris.data

import android.content.Context
import com.opencsv.CSVReader
import java.io.InputStreamReader

// Helper class to load data from given CSV file and transform it into a Array<FeatureColumn>
class DataFrame( context: Context , assetsFileName : String ) {

    // HashMap which stores the CSV data in the form ( Column_Name , Float[] ). Where Float[] holds
    // the feature value for all samples.
    private var featureColumnData = HashMap<String,ArrayList<Float>>()

    // Variable to store the parsed CSV file from `CSVReader`.
    private var rawData : List<Array<String>>

    // Variable to store column names which are rawData[0].
    private var columnNames : Array<String>

    // Will be called when the CSV file has been transformed to Array<FeatureColumn>.
    private lateinit var readCSVCallback : ReadCSVCallback


    // Number of features in the dataset. This number equals ( num_cols - 1 ) where num_cols is the number of
    // columns in the CSV file.
    // ( Note: We assume that the file has the labels column as the last column ).
    var numFeatures = 0

    // Number of samples in the dataset. This number equals ( num_rows - 1 ) where num_rows is the number of
    // rows in the CSV file.
    // Note: We assume that the CSV file has its first row as the column names.
    var numSamples = 0

    // Number of distinct elements in the "labels" column of the dataset.
    var numClasses = 0

    // Final output of the DataFrame class. This variable will be accessed by GaussianNB class.
    lateinit var featureColumns : Array<FeatureColumn>
    lateinit var labels : Array<String>


    init {
        // Create a CSVReader object
        val csvReader = CSVReader( InputStreamReader( context.assets.open( assetsFileName ) ) )

        // Call csvReader.readAll() which outputs the contents of the CSV in the form List<String[]>.
        // Every String[] corresponds to a single row.
        rawData = csvReader.readAll()

        // Get the column names.
        columnNames = getColumnNames()

        // Initialize variables like `numFeatures` and `numSamples`.
        // Also initialize `featureColumnData` by setting the keys as column names of this Hashmap.
        initTable()
    }

    // Parse the CSV file and call onCSVProcessed when done.
    fun readCSV( readCSVCallback: ReadCSVCallback) {
        this.readCSVCallback = readCSVCallback
        populateColumns()
    }

    // Convert the given String[] to a Float[]
    // Note: The `CSVReader` returns a row as String[] where each String is a number. We parse this String and convert it to
    // a float.
    fun convertStringArrayToFloatArray( strArray : Array<String> ) : FloatArray {
        val out = strArray.map { si -> si.toFloat() }.toFloatArray()
        // TODO : Check for exceptions
        return out
    }

    override fun toString(): String {
        return "DataFrame with $numFeatures feature(s) and $numSamples samples."
    }

    // Callback to check whether the given CSV file has been processed.
    interface ReadCSVCallback {
        fun onCSVProcessed(processedData : Array<FeatureColumn>, labels : Array<String> )
    }


    private fun getColumnNames() : Array<String> {
        val columnNames = rawData[ 0 ]
        return columnNames
    }

    // Set data from `rawData` to `featureColumnData`
    private fun populateColumns() {
        // Create an empty ArrayList to store the labels
        val labels = ArrayList<String>()

        // Iterate through `rawData` starting from index=1 ( as index=0 refers to the column names )
        for ( strSample in rawData.subList( 1 , rawData.size ) ) {

            // Append the label which is the last element of `strSample`.
            labels.add( strSample[ numFeatures ] )

            // Convert rest of the elements of `strSample` to FloatArray.
            val floatSample = convertStringArrayToFloatArray( strSample.sliceArray( IntRange( 0 , numFeatures - 1 ) ) )

            // Append each of the elements in above FloatArray to `featureColumnData`.
            floatSample.forEachIndexed { index, fl ->
                featureColumnData[ columnNames[ index ] ]!!.add( fl )
            }
        }

        // Transform `featureColumnData` HashMap into a Array<FeatureColumn>.
        val featureColumnsList = ArrayList<FeatureColumn>()
        for ( ( name , data ) in featureColumnData ) {
            featureColumnsList.add(
                FeatureColumn( name , data.toFloatArray() )
            )
        }
        featureColumns = featureColumnsList.toTypedArray()

        // Initialize `labels` here by getting the distinct elements from the labels column.
        this.labels = labels.toTypedArray()
        // num_classes = num_labels.
        numClasses = this.labels.distinct().size

        // We're done, call onCSVProcessed with `featureColumns` and `labels`.
        readCSVCallback.onCSVProcessed( featureColumns, this.labels )
    }

    // Initialize the `featureColumnData`
    private fun initTable() {
        // `numFeatures` = num_cols - 1
        numFeatures = rawData[ 0 ].size - 1

        // `numSamples` = num_rows - 1
        numSamples = rawData.size - 1

        // For each entry in `featureColumnData` initialize an empty ArrayList with key=column_name.
        for ( i in 0 until numFeatures ) {
            featureColumnData[ columnNames[ i ] ] = ArrayList()
        }
    }

}