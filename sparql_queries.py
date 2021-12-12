# Example 1: Given a NN and search in the knowledge graph which device is capable of running it
# In the example 1, we assume that based on the parsing result we have to feed accelerometer and gyroscope data
# into the NN. And the minimum RAM for running NN for RAM is 116 kb and Flash is 531 kb
query_1 = str("""
PREFIX s3n: <http://w3id.org/s3n/>
PREFIX sosa: <http://www.w3.org/ns/sosa/> 
PREFIX ssn: <http://www.w3.org/ns/ssn/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ssn-system: <http://www.w3.org/ns/ssn/systems/>
PREFIX schema: <https://schema.org/>
PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX td: <https://www.w3.org/2019/wot/td#>
PREFIX nnet: <http://tinyml-schema.org/networkschema#>
PREFIX s3n_extend: <http://tinyml-schema.org/s3n_extend/> 
PREFIX sosa_extend: <http://tinyml-schema.org/sosa_extend/> 
PREFIX ssn_extend: <http://tinyml-schema.org/ssn_extend/>

SELECT ?Board ?RAM ?Flash
WHERE {
    ?Board a s3n:SmartSensor ;
    	ssn:hasSubSystem ?system_1 ;
		ssn:hasSubSystem ?system_2 ;      	
		ssn:hasSubSystem ?system_3 .
		?system_1 a sosa_extend:Accelerometer .
		?system_2 a sosa_extend:Gyroscope .
		?system_3 a s3n:MicroController ;
			s3n:hasSystemCapability ?x .
		?x ssn-system:hasSystemProperty ?cond_1 .
        ?x ssn-system:hasSystemProperty ?cond_2 .
		?cond_1 a s3n_extend:RAM ;
			schema:value ?RAM ;
			schema:unitCode om:kilobyte .
    	?cond_2 a s3n_extend:Flash ;
			schema:value ?Flash ;
			schema:unitCode om:kilobyte .
		FILTER (?RAM >= 116)
    	FILTER (?Flash >= 531)
    }
""")

# Example 2: Given a microcontroller and search in the knowledge graph which NN can be used on the device
# In the example 2, we assume that based on the parsing result the microcontroller has a camera sensor
# And it has only is 127 kb free RAM and 576 kb free Flash
query_2 = str("""
PREFIX s3n: <http://w3id.org/s3n/>
PREFIX sosa: <http://www.w3.org/ns/sosa/> 
PREFIX ssn: <http://www.w3.org/ns/ssn/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ssn-system: <http://www.w3.org/ns/ssn/systems/>
PREFIX schema: <https://schema.org/>
PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX td: <https://www.w3.org/2019/wot/td#>
PREFIX nnet: <http://tinyml-schema.org/networkschema#>
PREFIX s3n_extend: <http://tinyml-schema.org/s3n_extend/> 
PREFIX sosa_extend: <http://tinyml-schema.org/sosa_extend/> 
PREFIX ssn_extend: <http://tinyml-schema.org/ssn_extend/>

SELECT ?nn ?uuid ?MACs ?RAM ?Flash
    WHERE {
        ?nn a nnet:NeuralNetwork ;
        schema:identifier ?uuid ;
        ssn:hasInput ?input;
        nnet:hasMultiplyAccumulateOps ?MACs ;
        s3n:hasProcedureFeature ?x_1 ;
        s3n:hasProcedureFeature ?x_2 .
    	?x_1 ssn-system:inCondition ?cond_1 .
        ?x_2 ssn-system:inCondition ?cond_2 .
        ?cond_1 a s3n_extend:RAM ;
    		schema:minValue ?RAM .
        ?cond_2 a s3n_extend:Flash ;
    		schema:minValue ?Flash .
    	?sensor ssn_extend:provideInput ?input;
             a sosa_extend:Camera .
        FILTER (?RAM <= 127)
        FILTER (?Flash <= 576)
    }
""")

# Example 3: Suppose we want to run a NN on an existing device for a specific task.
# We want to query and benchmark NN that can fulfill the requirements such as memory, sensor, task specification
query_3 = str("""
PREFIX s3n: <http://w3id.org/s3n/>
PREFIX sosa: <http://www.w3.org/ns/sosa/> 
PREFIX ssn: <http://www.w3.org/ns/ssn/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ssn-system: <http://www.w3.org/ns/ssn/systems/>
PREFIX schema: <https://schema.org/>
PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX td: <https://www.w3.org/2019/wot/td#>
PREFIX nnet: <http://tinyml-schema.org/networkschema#>
PREFIX s3n_extend: <http://tinyml-schema.org/s3n_extend/> 
PREFIX sosa_extend: <http://tinyml-schema.org/sosa_extend/> 
PREFIX ssn_extend: <http://tinyml-schema.org/ssn_extend/>

SELECT ?uuid ?Category ?Acc ?MACs ?Min_RAM ?Min_Flash
WHERE {
    ?nn a nnet:NeuralNetwork ;
        schema:identifier ?uuid ;
        schema:description ?description;
        ssn:hasInput ?input;
        nnet:hasCategory ?Category ;
        nnet:hasMetric ?metric ;
        nnet:hasMultiplyAccumulateOps ?MACs ;
        nnet:trainingDataset ?dataset ;
        s3n:hasProcedureFeature ?x_1 ;
        s3n:hasProcedureFeature ?x_2 .
    ?metric a nnet:Top_1_accuracy .
    ?metric nnet:hasMetricValue ?Acc .
    ?x_1 ssn-system:inCondition ?cond_1 .
    ?x_2 ssn-system:inCondition ?cond_2 .
    ?cond_1 a s3n_extend:RAM ;
            schema:minValue ?Min_RAM .
    ?cond_2 a s3n_extend:Flash ;
            schema:minValue ?Min_Flash .
    ?sensor ssn_extend:provideInput ?input;
            a sosa_extend:Microphone .
    FILTER regex(?description, "yes/no", "i")
    FILTER regex(str(?dataset), "speech_commands", "i")
}ORDER BY ?Acc
""")