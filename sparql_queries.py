# Example 1: Given a NN and search in the knowledge graph which device is capable of running it
# In the example 1, we assume that based on the parsing result we have to feed accelerometer and gyroscope data
# into the NN. And the minimum requirement for RAM is 116 kb and for Flash is 531 kb
query_1 = str("""
PREFIX s3n: <http://w3id.org/s3n/>
PREFIX sosa: <http://www.w3.org/ns/sosa/> 
PREFIX ssn: <http://www.w3.org/ns/ssn/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX demo: <http://ureasoner.org/tinyml-demo#>
PREFIX ssn-system: <http://www.w3.org/ns/ssn/systems/>
PREFIX schema: <https://schema.org/>
PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX td: <https://www.w3.org/2019/wot/td#>
PREFIX nnet: <http://ureasoner.org/networkschema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?multiSensor ?ram_memory ?flash_memory ?description
WHERE {
    ?multiSensor a s3n:SmartSensor ;
    	ssn:hasSubSystem ?system_1 ;
		ssn:hasSubSystem ?system_2 ;      	
		ssn:hasSubSystem ?system_3 ;
  		td:description ?description.
		?system_1 a demo:Accelerometer .
		?system_2 a demo:Gyroscope .
		?system_3 a s3n:MicroController ;
			s3n:hasSystemCapability ?x .
		?x ssn-system:hasSystemProperty ?cond_1 .
        ?x ssn-system:hasSystemProperty ?cond_2 .
		?cond_1 a demo:RAM ;
			schema:value ?ram_memory ;
			schema:unitCode om:kilobyte ;
    		rdfs:comment ?ram_comment .
    	?cond_2 a demo:Flash ;
			schema:value ?flash_memory ;
			schema:unitCode om:kilobyte ;
    		rdfs:comment ?flash_comment .
		FILTER (?ram_memory >= 116.5)
    	FILTER (?flash_memory >= 531.6)
}""")

# Example 2: Given a microcontroller and search in the knowledge graph which NN can be used on the device
# In the example 2, we assume that based on the parsing result the microcontroller has a camera sensor
# And it has only is 117 kb free RAM and 236 kb free Flash
query_2 = str("""
PREFIX s3n: <http://w3id.org/s3n/>
PREFIX sosa: <http://www.w3.org/ns/sosa/> 
PREFIX ssn: <http://www.w3.org/ns/ssn/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX demo: <http://ureasoner.org/tinyml-demo#>
PREFIX ssn-system: <http://www.w3.org/ns/ssn/systems/>
PREFIX schema: <https://schema.org/>
PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX td: <https://www.w3.org/2019/wot/td#>
PREFIX nnet: <http://ureasoner.org/networkschema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?neuralNetwork ?id ?category ?MultiplyAccumulateOps ?ram_memory ?flash_memory ?hardwareInfo ?creator ?description ?reference
WHERE {
    ?neuralNetwork a nnet:NeuralNetwork ;
    nnet:hasID ?id ;               
    nnet:hasCategory ?category ;
    nnet:hasDescription ?description ; 
    nnet:hasHardwareReference ?hardware ;
    nnet:creator ?creator ;
    nnet:reference ?reference;
    nnet:hasMultiplyAccumulateOps ?MultiplyAccumulateOps ;
    s3n:hasProcedureFeature ?x_1 ;
    s3n:hasProcedureFeature ?x_2 ;
    ssn:hasInput ?input ;
    ssn:hasOutput ?output .

    ?input nnet:hasInputInfo ?inputInfo .
    ?output nnet:hasOutputInfo ?outputInfo .
    
	?x_1 ssn-system:inCondition ?cond_1 .
    ?x_2 ssn-system:inCondition ?cond_2 .
    ?cond_1 a demo:RAM ;
		schema:minValue ?ram_memory ;
		schema:unitCode ?unit .
    ?cond_2 a demo:Flash ;
		schema:minValue ?flash_memory ;
		schema:unitCode ?unit .
    
    ?hardware a nnet:Camera ;
              nnet:hasHardwareInfo ?hardwareInfo .
    
    FILTER (?ram_memory <= 117)
    FILTER (?flash_memory <= 236)
}
""")

# Example 3: We want to query and benchmark NN given a few conditions
query_3 = str("""
PREFIX s3n: <http://w3id.org/s3n/>
PREFIX sosa: <http://www.w3.org/ns/sosa/> 
PREFIX ssn: <http://www.w3.org/ns/ssn/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX demo: <http://ureasoner.org/tinyml-demo#>
PREFIX ssn-system: <http://www.w3.org/ns/ssn/systems/>
PREFIX schema: <https://schema.org/>
PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX td: <https://www.w3.org/2019/wot/td#>
PREFIX nnet: <http://ureasoner.org/networkschema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?neuralNetwork ?id ?category ?top_1_acc ?MultiplyAccumulateOps ?min_ram_memory ?min_flash_memory ?hardwareInfo ?creator ?description ?reference
WHERE {
    ?neuralNetwork a nnet:NeuralNetwork ;
    nnet:hasID ?id ;               
    nnet:hasCategory ?category ;
    nnet:hasDescription ?description ; 
    nnet:hasHardwareReference ?hardware ;
    nnet:creator ?creator ;
    nnet:reference ?reference ;
    nnet:hasMetric ?metric ;
    nnet:hasMultiplyAccumulateOps ?MultiplyAccumulateOps ;
    s3n:hasProcedureFeature ?x_1 ;
    s3n:hasProcedureFeature ?x_2 ;
    ssn:hasInput ?input ;
    ssn:hasOutput ?output .

    ?metric a nnet:top_1_accuracy .
    ?metric nnet:hasMetricValue ?top_1_acc .
            
    ?input nnet:hasInputInfo ?inputInfo .
    ?output nnet:hasOutputInfo ?outputInfo .
    
	?x_1 ssn-system:inCondition ?cond_1 .
    ?x_2 ssn-system:inCondition ?cond_2 .
    ?cond_1 a demo:RAM ;
		schema:minValue ?min_ram_memory ;
		schema:unitCode ?unit .
    ?cond_2 a demo:Flash ;
		schema:minValue ?min_flash_memory ;
		schema:unitCode ?unit .
    
    ?hardware a nnet:Microphone ;
              nnet:hasHardwareInfo ?hardwareInfo .
    
    FILTER regex(?description, "yes/no", "i")
}
ORDER BY ?top_1_acc
""")