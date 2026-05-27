using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WindArea : MonoBehaviour
{
    List<Rigidbody> RigidbodiesInWindZone = new List<Rigidbody>();

    public float windStrength;
    private float minWindStrength = 5.8f;
    private float maxWindStrength = 6.2f;
    private float smoothness = 1.0f;
    private float frequency = 1.0f;
    private Vector3 currentWindDirection;
    private Vector3 targetWindDirection;
    private float transitionTimer = 0f;
    private float transitionDuration = 0.1f;

    private AudioSource windSound; // Reference to the AudioSource component
    private float maxVolume = 0.11f; // Maximum volume set to 1/3 of the full volume
    private float minVolume = 0.02f; // Adjusted minimum volume to maintain the relative difference

    private void Start()
    {
        windSound = GameObject.Find("WindSoundEffect").GetComponent<AudioSource>();
        windSound.clip = Resources.Load<AudioClip>("Audio/WindSoundEffect"); // Load the wind sound effect
        windSound.loop = true; // Ensure the audio source is set to loop
        windSound.Play(); // Start playing the wind sound
        transitionTimer = transitionDuration;
        StartNewTransition(); // Start the initial wind transition
    }

    private void FixedUpdate()
    {
        transitionTimer += Time.deltaTime;

        if (transitionTimer >= transitionDuration)
        {
            StartNewTransition();
        }

        if (RigidbodiesInWindZone.Count > 0)
        {
            currentWindDirection = Vector3.Slerp(currentWindDirection, targetWindDirection, Time.deltaTime / smoothness);
            ApplyWindForce(); // Apply wind force continuously
        }
        
        float targetVolume = (RigidbodiesInWindZone.Count > 0) ? maxVolume : minVolume; // Adjusted volume logic
        windSound.volume = Mathf.Lerp(windSound.volume, targetVolume, Time.deltaTime / 1.0f); // Smooth volume transition
    }

    private void OnTriggerEnter(Collider col)
    {
        Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
        if (objectRigid != null)
        {
            RigidbodiesInWindZone.Add(objectRigid);
        }
    }

    private void OnTriggerExit(Collider col)
    {
        Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
        if (objectRigid != null)
        {
            RigidbodiesInWindZone.Remove(objectRigid);
        }
    }

    private void StartNewTransition()
    {
        targetWindDirection = GenerateRandomWindDirection();
        transitionTimer = 0f;
    }

    private Vector3 GenerateRandomWindDirection()
    {
        float x = Mathf.PerlinNoise(Time.time * frequency, Random.Range(0f, 1000f)) * 2.0f - 1.0f;
        float y = Mathf.PerlinNoise(Random.Range(0f, 1000f), Time.time * frequency) * 2.0f - 1.0f;
        return new Vector3(x, 0f, y).normalized;
    }

    // private Vector3 GenerateRandomWindDirection()
    // {
    //     // Perlin noise gives a result in [0, 1], we shift it to [-0.2, 0.2]
    //     float x = Mathf.PerlinNoise(Time.time * frequency, Random.Range(0f, 1000f)) * 0.4f - 0.2f;
    //     float y = Mathf.PerlinNoise(Random.Range(0f, 1000f), Time.time * frequency) * 0.4f - 0.2f;
        
    //     // Debug log to check the actual values of x and y
    //     Debug.Log("Wind Direction: " + new Vector3(x, 0f, y));
        
    //     // Return the new Vector3 with the calculated x and y values
    //     return new Vector3(x, 0f, y);
    // }

    private void ApplyWindForce()
    {
        windStrength = Mathf.Lerp(minWindStrength, maxWindStrength, (Mathf.Sin(Time.time * frequency) + 1) / 2f);

        foreach (Rigidbody rigid in RigidbodiesInWindZone)
        {
            Vector3 windForce = currentWindDirection * windStrength;
            rigid.AddForce(windForce, ForceMode.Acceleration); // Apply force as acceleration
        }
    }
}






/////////////////////////////////////////////////////////////////////////////////////////
// using System.Collections.Generic;
// using UnityEngine;

// public class WindArea : MonoBehaviour
// {
//     List<Rigidbody> RigidbodiesInWindZone = new List<Rigidbody>();

//     public float windStrength;
//     private float minWindStrength = 5.8f; // Set your desired minimum wind strength
//     private float maxWindStrength = 6.2f; // Set your desired maximum wind strength
//     private float smoothness = 1.0f;
//     private float frequency = 1.0f;
//     private Vector3 currentWindDirection;
//     private Vector3 targetWindDirection;
//     private float transitionTimer = 0f;
//     private float transitionDuration = 0.1f;

//     private void Start()
//     {
//         transitionTimer = transitionDuration;
//         StartNewTransition(); // Start the initial wind transition
//     }

//     private void OnTriggerEnter(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Add(objectRigid);
//         }
//     }

//     private void OnTriggerExit(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Remove(objectRigid);
//         }
//     }

//     private void FixedUpdate()
//     {
//         transitionTimer += Time.deltaTime;

//         if (transitionTimer >= transitionDuration)
//         {
//             StartNewTransition();
//         }

//         if (RigidbodiesInWindZone.Count > 0)
//         {
//             currentWindDirection = Vector3.Slerp(currentWindDirection, targetWindDirection, Time.deltaTime / smoothness);
//             ApplyWindForce(); // Apply wind force continuously
//         }
//     }

//     private void StartNewTransition()
//     {
//         targetWindDirection = GenerateRandomWindDirection();
//         transitionTimer = 0f;
//     }

//     private Vector3 GenerateRandomWindDirection()
//     {
//         float x = Mathf.PerlinNoise(Time.time * frequency, Random.Range(0f, 1000f)) * 2.0f - 1.0f;
//         float y = Mathf.PerlinNoise(Random.Range(0f, 1000f), Time.time * frequency) * 2.0f - 1.0f;
//         // Debug.Log("Wind Direction: " + new Vector3(x, 0f, y).normalized);
//         return new Vector3(x, 0f, y).normalized;
//     }

//     // private Vector3 GenerateRandomWindDirection()
//     // {
//     //     // Perlin noise gives a result in [0, 1], we shift it to [-0.2, 0.2]
//     //     float x = Mathf.PerlinNoise(Time.time * frequency, Random.Range(0f, 1000f)) * 0.4f - 0.2f;
//     //     float y = Mathf.PerlinNoise(Random.Range(0f, 1000f), Time.time * frequency) * 0.4f - 0.2f;
        
//     //     // Debug log to check the actual values of x and y
//     //     Debug.Log("Wind Direction: " + new Vector3(x, 0f, y));
        
//     //     // Return the new Vector3 with the calculated x and y values
//     //     return new Vector3(x, 0f, y);
//     // }

//     private void ApplyWindForce()
//     {
//         windStrength = Mathf.Lerp(minWindStrength, maxWindStrength, (Mathf.Sin(Time.time * frequency) + 1) / 2f);

//         foreach (Rigidbody rigid in RigidbodiesInWindZone)
//         {
//             Vector3 windForce = currentWindDirection * windStrength;
//             rigid.AddForce(windForce, ForceMode.Acceleration); // Apply force as acceleration
//         }
//     }

// }




/////////////////////////////////////////////////////////////////////////////////
// Applying Wind Force with transition timer

// using System.Collections.Generic;
// using UnityEngine;

// public class WindArea : MonoBehaviour
// {
//     List<Rigidbody> RigidbodiesInWindZone = new List<Rigidbody>();

//     public float windStrength;
//     private float minWindStrength = 290f; // Set your desired minimum wind strength
//     private float maxWindStrength = 310f; // Set your desired maximum wind strength
//     private float smoothness = 1.0f;
//     private float frequency = 1.0f;
//     private Vector3 currentWindDirection;
//     private Vector3 targetWindDirection;
//     private float transitionTimer = 0f;
//     private float transitionDuration = 10f;
//     private float windForceTimer = 0f;
//     private float windForceInterval = 0.05f; // Adjusted to apply wind force every 0.5 seconds

//     private void Start()
//     {
//         transitionTimer = transitionDuration;
//         windForceTimer = windForceInterval; // Set the timer to apply wind force immediately
//         StartNewTransition(); // Start the initial wind transition
//     }

//     private void OnTriggerEnter(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Add(objectRigid);
//         }
//     }

//     private void OnTriggerExit(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Remove(objectRigid);
//         }
//     }

//     private void Update()
//     {
//         transitionTimer += Time.deltaTime;
//         windForceTimer += Time.deltaTime;

//         if (transitionTimer >= transitionDuration)
//         {
//             // Start a new transition
//             StartNewTransition();
//         }

//         if (windForceTimer >= windForceInterval)
//         {
//             // Apply wind force every 5 seconds
//             ApplyWindForce();
//         }

//         if (RigidbodiesInWindZone.Count > 0)
//         {
//             // Smoothly interpolate between current and target wind directions
//             currentWindDirection = Vector3.Slerp(currentWindDirection, targetWindDirection, Time.deltaTime / smoothness);
//         }
//     }

//     private void StartNewTransition()
//     {
//         // Set a new random target wind direction
//         targetWindDirection = GenerateRandomWindDirection();

//         // Reset the transition timer
//         transitionTimer = 0f;
//     }

//     private Vector3 GenerateRandomWindDirection()
//     {
//         // Use Perlin noise to generate smooth random values for wind direction
//         float x = Mathf.PerlinNoise(Time.time * frequency, Random.Range(0f, 1000f)) * 2.0f - 1.0f;
//         float y = Mathf.PerlinNoise(Random.Range(0f, 1000f), Time.time * frequency) * 2.0f - 1.0f;

//         // Create a smooth random direction vector
//         return new Vector3(x, 0f, y).normalized;
//     }

//     private void ApplyWindForce()
//     {
//         windStrength = Mathf.Lerp(minWindStrength, maxWindStrength, (Mathf.Sin(Time.time * frequency) + 1) / 2f);
        
//         foreach (Rigidbody rigid in RigidbodiesInWindZone)
//         {
//             // Calculate the wind force
//             Vector3 windForce = currentWindDirection * windStrength;

//             // Apply the wind force to the rigidbody
//             rigid.AddForce(windForce);

//             // Debug log to check which objects are affected and what force is applied
//             Debug.Log("Wind affecting " + rigid.gameObject.name + " with force: " + windForce);
//         }

//         // Reset the wind force timer
//         windForceTimer = 0f;
//     }
// }




/////////////////////////////////////////////////////////////////////////////////
// using System.Collections.Generic;
// using UnityEngine;

// public class WindArea : MonoBehaviour
// {
//     List<Rigidbody> RigidbodiesInWindZone = new List<Rigidbody>();

//     public float windStrength;
//     private float minWindStrength = 29f; // Set your desired minimum wind strength
//     private float maxWindStrength = 31f; // Set your desired maximum wind strength
//     private float smoothness = 1.0f;
//     private float frequency = 1.0f;
//     private Vector3 currentWindDirection;
//     private Vector3 targetWindDirection;
//     private float transitionTimer = 0f;
//     private float transitionDuration = 10f;
//     private float windForceTimer = 0f;
//     private float windForceInterval = 5f; // Adjusted to apply wind force every 5 seconds

//     private void Start()
//     {
//         transitionTimer = transitionDuration;
//         windForceTimer = windForceInterval; // Set the timer to apply wind force immediately
//         StartNewTransition(); // Start the initial wind transition
//     }

//     private void OnTriggerEnter(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Add(objectRigid);
//         }
//     }

//     private void OnTriggerExit(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Remove(objectRigid);
//         }
//     }

//     private void Update()
//     {
//         transitionTimer += Time.deltaTime;
//         windForceTimer += Time.deltaTime;

//         if (transitionTimer >= transitionDuration)
//         {
//             // Start a new transition
//             StartNewTransition();
//         }

//         if (windForceTimer >= windForceInterval)
//         {
//             // Apply wind force every 5 seconds
//             ApplyWindForce();
//         }

//         if (RigidbodiesInWindZone.Count > 0)
//         {
//             // Smoothly interpolate between current and target wind directions
//             currentWindDirection = Vector3.Slerp(currentWindDirection, targetWindDirection, Time.deltaTime / smoothness);
//         }
//     }

//     private void StartNewTransition()
//     {
//         // Set a new random target wind direction
//         targetWindDirection = GenerateRandomWindDirection();

//         // Reset the transition timer
//         transitionTimer = 0f;
//     }

//     private Vector3 GenerateRandomWindDirection()
//     {
//         // Use Perlin noise to generate smooth random values for wind direction
//         float x = Mathf.PerlinNoise(Time.time * frequency, Random.Range(0f, 1000f)) * 2.0f - 1.0f;
//         float y = Mathf.PerlinNoise(Random.Range(0f, 1000f), Time.time * frequency) * 2.0f - 1.0f;

//         // Create a smooth random direction vector
//         return new Vector3(x, 0f, y).normalized;
//     }

//     private void ApplyWindForce()
//     {
//         windStrength = Mathf.Lerp(minWindStrength, maxWindStrength, (Mathf.Sin(Time.time * frequency) + 1) / 2f);
        
//         foreach (Rigidbody rigid in RigidbodiesInWindZone)
//         {
//             // Apply the current wind direction to each rigidbody's wind force
//             rigid.AddForce(currentWindDirection * windStrength);
//         }

//         // Reset the wind force timer
//         windForceTimer = 0f;
//     }
// }
////////////////////////////////////////////////////////////////////////////

// using System.Collections;
// using System.Collections.Generic;
// using Unity.VisualScripting;
// using UnityEngine;

// public class WindArea : MonoBehaviour
// {
//     List<Rigidbody> RigidbodiesInWindZone = new List<Rigidbody>();

//     public float windStrength = 20f;
//     public float smoothness = 1.0f;
//     public float frequency = 1.0f;
//     private Vector3 currentWindDirection;
//     private Vector3 targetWindDirection;
//     private float transitionTimer = 0f;
//     public float transitionDuration = 10f;

//     private void Start()
//     {
//         transitionTimer = transitionDuration;
//     }

//     private void OnTriggerEnter(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Add(objectRigid);
//         }
//     }

//     private void OnTriggerExit(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Remove(objectRigid);
//         }
//     }

//     private void Update()
//     {
//         transitionTimer += Time.deltaTime;

//         if (transitionTimer >= transitionDuration)
//         {
//             // Start a new transition
//             StartNewTransition();
//         }


//         if (RigidbodiesInWindZone.Count > 0)
//         {
//             // Smoothly interpolate between current and target wind directions
//             currentWindDirection = Vector3.Slerp(currentWindDirection, targetWindDirection, Time.deltaTime / smoothness);

//             foreach (Rigidbody rigid in RigidbodiesInWindZone)
//             {
//                 // Apply the current wind direction to each rigidbody's wind force
//                 rigid.AddForce(currentWindDirection * windStrength);
//             }
//         }
//     }

//     private void StartNewTransition()
//     {
//         // Set a new random target wind direction
//         targetWindDirection = GenerateRandomWindDirection();

//         // Reset the transition timer
//         transitionTimer = 0f;
//     }

//     private Vector3 GenerateRandomWindDirection()
//     {
//         // Use Perlin noise to generate smooth random values for wind direction
//         float x = Mathf.PerlinNoise(Time.time * frequency, Random.Range(0f, 1000f)) * 2.0f - 1.0f;
//         float y = Mathf.PerlinNoise(Random.Range(0f, 1000f), Time.time * frequency) * 2.0f - 1.0f;

//         // Create a smooth random direction vector
//         return new Vector3(x, 0f, y).normalized;
//     }
// }



// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;

// public class WindArea : MonoBehaviour
// {
//     List<Rigidbody> RigidbodiesInWindZone = new List<Rigidbody>();

//     public float windStrength;
//     public float smoothness = 1.0f;
//     public float frequency = 1.0f;
//     private Quaternion currentWindRotation;
//     private Quaternion targetWindRotation;
//     private float transitionTimer = 0f;
//     public float transitionDuration = 10f;

//     private void OnTriggerEnter(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Add(objectRigid);
//         }
//     }

//     private void OnTriggerExit(Collider col)
//     {
//         Rigidbody objectRigid = col.gameObject.GetComponent<Rigidbody>();
//         if (objectRigid != null)
//         {
//             RigidbodiesInWindZone.Remove(objectRigid);
//         }
//     }

//     private void Start()
//     {
//         transitionTimer = transitionDuration;
//     }

//     private void Update()
//     {
//         transitionTimer += Time.deltaTime;

//         if (transitionTimer >= transitionDuration)
//         {
//             // Start a new transition
//             StartNewTransition();
//         }

//         if (RigidbodiesInWindZone.Count > 0)
//         {
//             // Smoothly interpolate between current and target wind directions
//             currentWindRotation = Quaternion.Slerp(currentWindRotation, targetWindRotation, Time.deltaTime / smoothness);

//             foreach (Rigidbody rigid in RigidbodiesInWindZone)
//             {
//                 // Apply the current wind direction to each rigidbody's wind force
//                 rigid.AddForce(currentWindRotation * Vector3.forward * windStrength);
//             }
//         }
//     }

//     private void StartNewTransition()
//     {
//         // Set a new random target wind direction
//         targetWindRotation = GenerateRandomWindRotation();

//         // Reset the transition timer
//         transitionTimer = 0f;
//     }

//     private Quaternion GenerateRandomWindRotation()
//     {
//         // Use Perlin noise to generate smooth random values for wind direction
//         float x = Mathf.PerlinNoise(Time.time * frequency, Random.Range(0f, 1000f)) * 2.0f - 1.0f;
//         float y = Mathf.PerlinNoise(Random.Range(0f, 1000f), Time.time * frequency) * 2.0f - 1.0f;

//         // Create a smooth random direction vector
//         Vector3 randomDirection = new Vector3(x, 0f, y).normalized;

//         // Convert the vector to a quaternion for smooth rotation
//         return Quaternion.LookRotation(randomDirection, Vector3.up);
//     }
// }