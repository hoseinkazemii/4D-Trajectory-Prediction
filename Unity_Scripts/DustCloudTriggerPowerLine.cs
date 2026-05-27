using UnityEngine;
using System;

public class DustCloudTriggerPowerLine : MonoBehaviour
{
    public ParticleSystem dustCloudPrefab;
    private Transform grabbableObject;
    public GameObject craneHook;
    public static float taskEndTime;

    private float proximityThreshold = 0.5f; // Adjust this value to detect proximity
    public static bool taskEnded = false;

    public CSVWriter csvWriter;
    public VREyeTrackingDataManagerPowerLine vrEyeTrackerPowerLine; // Add reference to the VR eye tracker

    // Event to send collision data
    public static event Action<string, float, Vector3> OnCollisionLogged;

    void Start()
    {
        grabbableObject = transform;
    }

    void Update()
    {
        if (!grabbableObject.transform.IsChildOf(craneHook.transform))
        {
            // Check if grabbable object is close enough to the UnloadingStorey or UnloadingArea
            Collider[] colliders = Physics.OverlapSphere(grabbableObject.position, proximityThreshold);
            foreach (Collider collider in colliders)
            {
                if ((collider.gameObject.layer == LayerMask.NameToLayer("UnloadingStorey") || 
                    collider.gameObject.layer == LayerMask.NameToLayer("UnloadingArea")) && !taskEnded)
                {
                    RecordTaskEndTime();
                    taskEnded = true;
                    csvWriter.StopWriting(); // Stop writing the load movement (signals) data when the task ends
                    vrEyeTrackerPowerLine.StopEyeTracking(); // Stop writing VR eye tracking data when the task ends
                }
            }
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if ((collision.gameObject.layer == LayerMask.NameToLayer("UnloadingStorey") || collision.gameObject.layer == LayerMask.NameToLayer("UnloadingArea"))
            && !grabbableObject.transform.IsChildOf(craneHook.transform))
        {
            RecordTaskEndTime();
            // Instantiate dust cloud at the final position
            if (dustCloudPrefab != null)
            {
                ParticleSystem dustCloud = Instantiate(dustCloudPrefab, transform.position, Quaternion.identity);
                dustCloud.Play();
            }
        }
        // Log the collision event with name, time, and coordinates
        OnCollisionLogged?.Invoke(collision.gameObject.name, Time.time, grabbableObject.position);
    }

    private void RecordTaskEndTime()
    {
        if (!taskEnded)
        {
            taskEndTime = Time.time;
        }
    }
}





// using UnityEngine;

// public class DustCloudTrigger : MonoBehaviour
// {
//     public ParticleSystem dustCloudPrefab;
//     private Transform grabbableObject;
//     public GameObject craneHook;
//     public static float taskEndTime;

//     void Start()
//     {
//         grabbableObject = transform;
//     }

//     private void OnCollisionEnter(Collision collision)
//     {
//         if ((collision.gameObject.layer == LayerMask.NameToLayer("UnloadingStorey") || collision.gameObject.layer == LayerMask.NameToLayer("UnloadingArea"))
//             && !grabbableObject.transform.IsChildOf(craneHook.transform))
//         {
//             taskEndTime = Time.time;

//             if (dustCloudPrefab != null)
//             {
//                 ParticleSystem dustCloud = Instantiate(dustCloudPrefab, transform.position, Quaternion.identity);
//                 dustCloud.Play();
//             }
//         }
//     }
// }