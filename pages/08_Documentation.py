import streamlit as  st

st.markdown('''

## Sample Global Query

```             
$     python -m graphrag.query \
>     --root ./data \
>     --method global \
>     "How many heat related accidents were there?"
```

```
INFO: Reading settings from data\settings.yaml
creating llm client with {'api_key': 'REDACTED,len=56', 'type': "openai_chat", 'model': 'gpt-4-turbo-preview', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'n': 1, 'request_timeout': 180.0, 'api_base': None, 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}
SUCCESS: Global Search Response:
```
            

### Overview of Heat-Related Incidents

Recent reports have highlighted a concerning trend of heat-related incidents across multiple plants, underscoring the urgent need for enhanced safety measures and employee training. These incidents primarily involve cases of heat exhaustion among workers, including contractors, during their duties. The data reveals that Plants A, B, and C have all experienced such safety concerns, with varying degrees of severity and implications for workplace safety protocols.

### Detailed Incident Reports

- **Plant A**: There has been at least one documented incident involving a contractor experiencing symptoms of heat exhaustion while performing roof repairs [Data: Reports (5)]. This incident highlights the risks faced by workers, especially those involved in physically demanding tasks in exposed conditions.

- **Plant B**: Workers at this plant have been directly affected by heat exhaustion, indicating a broader issue with the working conditions and the effectiveness of current safety protocols [Data: Reports (12)]. The recurrence of such incidents at Plant B suggests that existing measures may be insufficient or inadequately enforced, necessitating a thorough review and enhancement of safety practices.

- **Plant C**: Similar to Plants A and B, Plant C has encountered safety incidents, including cases of heat exhaustion among contractors [Data: Reports (6)]. This underscores a widespread challenge across the plants in managing the risks associated with high temperatures and strenuous work environments.

### Implications and Recommendations

The occurrence of heat-related incidents across Plants A, B, and C calls for immediate action to prevent further occurrences. It is imperative that all plants review and strengthen their safety protocols, focusing on measures to mitigate heat exposure and improve the overall working conditions. This may include enhanced training for employees on recognizing and responding to symptoms of heat-related illnesses, provision of adequate hydration and rest breaks, and the implementation of more rigorous monitoring during high-temperature conditions.

In conclusion, the reported incidents of heat exhaustion across the plants serve as a critical reminder of the importance of workplace safety and the need for continuous improvement in safety measures. Addressing these issues will not only help in preventing future incidents but also contribute to creating a safer and more productive working environment for all employees.
            

            
## Sample Local Query
```
$     python -m graphrag.query \
>     --root ./data \ 
>     --method local \
>     "Which events included fire?"
```

## Response
```            
INFO: Reading settings from data\settings.yaml

INFO: Vector Store Args: {}
[2024-09-04T19:37:57Z WARN  lance::dataset] No existing dataset at C:\dev\graphrag-demo\data\output\20240904-223419\artifacts\lancedb\entity_description_embeddings.lance, it will be created
creating llm client with {'api_key': 'REDACTED,len=56', 'type': "openai_chat", 'model': 'gpt-4-turbo-preview', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'n': 1, 'request_timeout': 180.0, 'api_base': None, 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}
creating embedding llm client with {'api_key': 'REDACTED,len=56', 'type': "openai_embedding", 'model': 'text-embedding-3-small', 'max_tokens': 4000, 'temperature': 0, 'top_p': 1, 'n': 1, 'request_timeout': 180.0, 'api_base': None, 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': None, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}
```   
            
### SUCCESS: Local Search Response:

The events that included fire or fire-related incidents, specifically involving the activation of fire alarms and sprinkler systems due to overheated equipment, occurred across multiple plants. These 
incidents highlight the importance of maintaining equipment and adhering to safety protocols to prevent potential fire hazards. Below is a summary of the fire-related events:

### Plant B Fire-Related Incidents
- **Overheated Boiler and Conveyor Belt**: On several occasions, the overheating of boilers and a conveyor belt triggered the fire alarm system, leading to the activation of the sprinkler system. These incidents caused minimal water damage and production halts ranging from 1 to 7 hours [Data: Sources (79, 5, 30, 87, 60)].
- **Dates of Incidents**: The incidents occurred on various dates throughout 2024, including 2024-07-19, 2024-09-15, 2024-04-21, and 2024-09-14, among others [Data: Entities (53, 14, 37, +more)].     

### Plant C Fire-Related Incidents
- **Overheated Boiler and Machines**: Similar to Plant B, Plant C experienced fire alarms triggered by overheated boilers and machines. These incidents also led to the activation of the sprinkler system, causing minimal water damage and temporary production stops [Data: Sources (58, 57, 32, 55)].
- **Dates of Incidents**: These events were reported on dates including 2024-08-18, 2024-06-17, 2024-08-11, and 2024-08-24, highlighting a recurring issue with equipment overheating [Data: Entities (47, 46, 39, 45)].

### Plant A Fire-Related Incident
- **Overheated Machine**: Plant A had a fire alarm triggered by an overheated machine, which also resulted in the sprinkler system being activated. This incident caused minimal water damage and a brief production halt [Data: Source (65)].
- **Date of Incident**: This particular event took place on 2024-06-04 [Data: Entity (50)].

These incidents across the plants underscore the critical need for regular equipment maintenance and safety checks to prevent overheating and potential fires. The activation of sprinkler systems, while effective in minimizing damage, also points to the necessity of having robust fire prevention and response measures in place.

            
''')